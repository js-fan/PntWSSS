import torch
import torch.nn as nn
import torch.nn.functional as F
from core.utils import *
from .memory_helper import *

def _cross_entropy_2d(logit, label, real_num=True):
    if real_num:
        loss = F.cross_entropy(logit, label, ignore_index=255, reduction='sum')
        loss = loss / torch.clamp_min((label < 255).sum(), 1)
    else:
        loss = F.cross_entropy(logit, label, ignore_index=255)
    return loss

def _contrastive(x, ref, pmask, nmask, tau=0.07):
    # x:   [N, D, S]
    # ref: [N, D, T]
    # pmask, nmask: [N, S, T]
    dot = (x.transpose(1, 2) @ ref) / tau # [N, S, T]

    exists = torch.minimum(pmask.max(2)[0], nmask.max(2)[0]) # [N, S]
    mask = (1 - torch.maximum(pmask, nmask)) * exists.unsqueeze(2) # [N, S, T]
    dot.masked_fill_(mask.to(bool), float('-inf'))
    
    log_softmax = torch.log(torch.softmax(dot, 2) + 1e-5)
    loss = -(log_softmax * pmask).sum(2) / torch.clamp_min(pmask.sum(2), 1) # [N, S]
    loss = loss * exists
    loss_red = (loss.sum(1) / torch.clamp_min(exists.sum(1), 1)).mean()
    return loss_red

def _anti_eye(n, dtype=torch.float32, device='cpu'):
    return 1 - torch.eye(n, dtype=dtype, device=device)

def _get_pix_proj(proj, sp_lbl2d, C, **kwargs):
    proj = F.normalize(proj, dim=1).flatten(2) # [N, D, HW]
    proj_lbl_oh = F.one_hot(sp_lbl2d.flatten(1), 256)[..., :C].float() # [N, HW, C]
    return proj, proj_lbl_oh

def _get_sp_proj(proj, sp_prob, sp_mask, C, threshold, **kwargs):
    sp_proj = proj.flatten(2) @ sp_mask.flatten(2).transpose(1, 2) # [N, D, M]
    sp_proj = F.normalize(sp_proj, dim=1)
    with torch.no_grad():
        sp_conf, sp_lbl = sp_prob.max(1) # [N, M]
        sp_lbl[sp_conf < threshold] = 255
        sp_lbl_oh = F.one_hot(sp_lbl, 256)[..., :C].float() # [N, M, C]
    return sp_proj, sp_lbl_oh
 
def _get_pnormimg_proj(proj, sp_lbl2d, C, **kwargs):
    N, H, W = sp_lbl2d.shape
    sp_lbl2d_oh = F.one_hot(sp_lbl2d, 256)[..., :C].float().view(N, H*W, C)
    proj = F.normalize(proj, dim=1)
    img_proj = proj.flatten(2) @ sp_lbl2d_oh.flatten(2) # [N, D, C]
    img_proj = img_proj / torch.clamp_min(sp_lbl2d_oh.sum(1, keepdim=True), 1)
    #img_proj = F.normalize(img_proj, dim=1)
    with torch.no_grad():
        sp_lbl2d_existing = sp_lbl2d_oh.max(1, keepdim=True)[0] # [N, 1, C]
        img_proj_lbl = sp_lbl2d_existing.transpose(1, 2) @ sp_lbl2d_existing # [N, C, C]
    return img_proj, img_proj_lbl

def _get_img_proj(proj, sp_lbl2d, C, **kwargs):
    N, H, W = sp_lbl2d.shape
    sp_lbl2d_oh = F.one_hot(sp_lbl2d, 256)[..., :C].float().view(N, H*W, C)
    img_proj = proj.flatten(2) @ sp_lbl2d_oh.flatten(2) # [N, D, C]
    img_proj = img_proj / torch.clamp_min(sp_lbl2d_oh.sum(1, keepdim=True), 1)
    img_proj = F.normalize(img_proj, dim=1)
    with torch.no_grad():
        sp_lbl2d_existing = sp_lbl2d_oh.max(1, keepdim=True)[0] # [N, 1, C]
        img_proj_lbl = sp_lbl2d_existing.transpose(1, 2) @ sp_lbl2d_existing # [N, C, C]
    return img_proj, img_proj_lbl

def _get_projections(proj_type, **kwargs):
    proj_src_fn, proj_tgt_fn = proj_type.split('2')
    self_proj = proj_src_fn == proj_tgt_fn
    proj_src_fn = eval(f"_get_{proj_src_fn}_proj")
    proj_tgt_fn = eval(f"_get_{proj_tgt_fn}_proj")
    proj_src, proj_src_lbl_oh = proj_src_fn(**kwargs)
    proj_tgt, proj_tgt_lbl_oh = proj_tgt_fn(**kwargs)
    return proj_src, proj_src_lbl_oh, proj_tgt, proj_tgt_lbl_oh, self_proj

def point_cl_universal(rank, mod, data, niter, cfg):
    if cfg.data.train.get('rtn_image_label', False):
        image, superpixel, label, existing_pnt_labels = data
    else:
        image, superpixel, label = data
    # image, label, superpixel, existing_pnt_labels = data
    logit, proj = mod(image.to(rank))
    label = torch_resize(label.to(rank), logit)
    loss_pnt = _cross_entropy_2d(logit, label)

    if not cfg.data.train.get('rtn_image_label', False):
        existing_pnt_labels = F.one_hot(label, 256)[..., :logit.shape[1]].float()
        existing_pnt_labels = existing_pnt_labels.max(2)[0].max(1)[0] # [N, C]
    existing_pnt_labels = existing_pnt_labels.to(rank)

    with torch.no_grad():
        sp_mask = F.one_hot(superpixel.to(rank), cfg.data.train.max_superpixel+1)
        sp_mask = sp_mask[..., :cfg.data.train.max_superpixel].permute(0,3,1,2)
        sp_mask = torch_resize(sp_mask.float(), logit).contiguous() # [N, M, H, W]

        prob2d = torch.softmax(logit, 1)
        sp_prob = prob2d.flatten(2) @ sp_mask.flatten(2).transpose(1,2) # [N, C, M]
        sp_prob = sp_prob / torch.clamp_min(sp_mask.sum((2,3)).unsqueeze(1), 1e-5)
        # sp_logit = logit.flatten(2) @ sp_mask.flatten(2).transpose(1,2) # [N, C, M]
        # sp_logit = sp_logit / torch.clamp_min(sp_mask.sum((2, 3)).unsqueeze(1), 1e-5)

        # sp_prob = torch.softmax(sp_logit, 1) # [N, C, M]
        sp_prob = sp_prob * existing_pnt_labels.unsqueeze(2)
        sp_prob2d = (sp_prob @ sp_mask.flatten(2)).view(logit.shape) 
        sp_conf2d, sp_lbl2d = sp_prob2d.max(1)
        sp_lbl2d[sp_conf2d < cfg.hyperparam.superpixel_threshold] = 255
    loss_sp = _cross_entropy_2d(logit, sp_lbl2d, False)

    # [N, D, S], [N, S, C], [N, D, T], [N, T, C]
    psrc, psrc_lbl_oh, ptgt, ptgt_lbl_oh, self_proj = _get_projections(
            cfg.hyperparam.proj_type,
            proj=proj,
            sp_lbl2d=sp_lbl2d,
            sp_prob=sp_prob,
            sp_mask=sp_mask,
            C=logit.shape[1],
            threshold=cfg.hyperparam.superpixel_threshold)

    # batch-wise
    if cfg.hyperparam.batch_wise:
        N, D, T = ptgt.shape
        ptgt = ptgt.transpose(0, 1).reshape(1, D, N*T).expand(N, -1, -1)
        ptgt_lbl_oh = ptgt_lbl_oh.view(1, N*T, logit.shape[1]).expand(N, -1, -1)
        self_proj = self_proj and (N == 1)

    # Memory
    if cfg.hyperparam.use_memory:
        if niter == 0:
            cfg._memer = TorchMemer(logit.shape[1], cfg.hyperparam.capacity)
        memer = cfg._memer

        with torch.no_grad():
            if cfg.hyperparam.batch_wise:
                ptgt = ptgt[0]
                ptgt_lbl_oh = ptgt_lbl_oh[0]
            else:
                N, D, T = ptgt.shape
                ptgt = ptgt.transpose(0, 1).view(D, N*T)
                ptgt_lbl_oh = ptgt_lbl_oh.view(N*T, logit.shape[1])
            ptgt = ptgt.transpose(0, 1).contiguous()
            memer.push(ptgt.data, ptgt_lbl_oh.data)

            ptgt_mem, ptgt_lbl_mem = memer.pull()
            atLeast = max(cfg.hyperparam.get('limit_pos', 0), 1)
            if ptgt_mem is None:
                N, D, S = psrc.shape
                ptgt = torch.zeros((N, D, atLeast), dtype=torch.float32, device=psrc.device)
                ptgt_lbl_oh = torch.zeros((N, atLeast, logit.shape[1]), dtype=torch.float32, device=psrc.device)
            else:
                ptgt = ptgt_mem.transpose(0, 1).unsqueeze(0).contiguous().expand(N, -1, -1) # [N, D, T]
                ptgt_lbl_oh = F.one_hot(ptgt_lbl_mem, logit.shape[1]).float().unsqueeze(0).expand(N, -1, -1) # [N, T, C]
            if ptgt.shape[2] < atLeast:
                N, D, T = ptgt.shape
                ptgt = torch.cat([ptgt, torch.zeros((N,D,atLeast-T), dtype=ptgt.dtype, device=ptgt.device)], 2)
                ptgt_lbl_oh = torch.cat([ptgt_lbl_oh, torch.zeros((N,atLeast-T,logit.shape[1]), dtype=ptgt_lbl_oh.dtype, device=ptgt_lbl_oh.device)], 1)

    with torch.no_grad():
        pmask = psrc_lbl_oh @ ptgt_lbl_oh.transpose(1, 2) # [N, S, T]
        nmask = (1 - pmask) * psrc_lbl_oh.max(2, keepdim=True)[0]
        if self_proj:
            pmask = pmask * _anti_eye(pmask.shape[1], pmask.dtype, pmask.device)

        # Limiting to N pos
        nPos = cfg.hyperparam.get('limit_pos', 0)
        if nPos > 0:
            pmask_priority = pmask * torch.rand(pmask.shape, dtype=torch.float32, device=pmask.device)
            if nPos == 1:
                pmask_selected = F.one_hot(pmask_priority.argmax(2), pmask.shape[2]).float()
            else:
                pmask_selected = torch.topk(pmask_priority, nPos, dim=2)[1]
                pmask_selected = F.one_hot(pmask_selected, pmask.shape[2]).float().max(2)[0]
            pmask = pmask * pmask_selected

    loss_cl = _contrastive(psrc, ptgt, pmask, nmask, cfg.hyperparam.tau)

    #
    if cfg.hyperparam.warmup_type == 'linear':
        warmup = min(float(niter) / (cfg.hyperparam.warmup * cfg.optimizer.niters_per_epoch), 1)
    elif cfg.hyperparam.warmup_type == 'exponential':
        warmup = min(float(niter) / (cfg.hyperparam.warmup * cfg.optimizer.niters_per_epoch), 1)
        warmup = np.exp((warmup - 1) * 5)
    else:
        raise RuntimeError(cfg.hyperparam.warmup_type)
    loss = loss_pnt + \
           loss_sp * (cfg.hyperparam.lambda_superpixel * warmup) + \
           loss_cl * (cfg.hyperparam.lambda_contrastive* warmup)

    nPos = pmask.sum() / torch.clamp_min(pmask.max(2)[0].sum(), 1)
    nNeg = nmask.sum() / torch.clamp_min(nmask.max(2)[0].sum(), 1)
    monitor_vals = {
            'loss_pnt': loss_pnt,
            'loss_sp': loss_sp,
            'loss_cl': loss_cl,
            'nPos': nPos,
            'nNeg': nNeg}
    monitor_imgs = [image[0], label[0], superpixel[0], sp_lbl2d[0], logit[0]]
    return loss, monitor_vals, monitor_imgs

