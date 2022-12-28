import torch
import torch.nn as nn
import torch.nn.functional as F
from core.utils import *
from .memory_helper import *
from core.model.myqueue import ClasswiseQueue

def _focal_contrastive_loss(src, tgt, pmask, nmask, tau=0.07, power=2):
    '''
    src: [N, D, S]
    tgt: [N, D, T]
    pmask, nmask: [N, S, T]
    '''
    dot = (src.transpose(1, 2) @ tgt) / tau # [N, S, T]
    relevent_elements = torch.maximum(pmask, nmask)
    valid_queries = pmask.max(2)[0] * nmask.max(2)[0]
    ignore_mask = (1 - relevent_elements) * valid_queries.unsqueeze(2)
    dot.masked_fill_(ignore_mask.to(bool), float('-inf'))

    softmax = torch.softmax(dot, 2)
    log_softmax = torch.log(torch.clamp_min(softmax, 1e-5))
    focal_weight = ((1 - softmax) * pmask)**power
    loss = -(log_softmax * pmask * focal_weight).sum(2) / torch.clamp_min(pmask.sum(2), 1e-3) * valid_queries
    loss_red = loss.sum(1) / torch.clamp_min(valid_queries.sum(1), 1e-3)
    loss_red2 = loss_red.mean()
    return loss_red2

def _contrastive_loss(src, tgt, pmask, nmask, tau=0.07):
    '''
    src: [N, D, S]
    tgt: [N, D, T]
    pmask, nmask: [N, S, T]
    '''
    dot = (src.transpose(1, 2) @ tgt) / tau # [N, S, T]
    relevent_elements = torch.maximum(pmask, nmask)
    valid_queries = pmask.max(2)[0] * nmask.max(2)[0]
    ignore_mask = (1 - relevent_elements) * valid_queries.unsqueeze(2)
    dot.masked_fill_(ignore_mask.to(bool), float('-inf'))

    log_softmax = torch.log(torch.clamp_min(torch.softmax(dot, 2), 1e-5))
    loss = -(log_softmax * pmask).sum(2) / torch.clamp_min(pmask.sum(2), 1e-3) * valid_queries
    loss_red = loss.sum(1) / torch.clamp_min(valid_queries.sum(1), 1e-3)
    loss_red2 = loss_red.mean()
    return loss_red2
    # pmask = pmask * valid_queries.unsqueeze(2)
    # loss = -(log_softmax * pmask).sum() / torch.clamp_min(pmask.sum(), 1)
    # return loss
    '''
    dot = (src.transpose(1, 2) @ tgt)
    dot_exp = torch.exp(dot / 0.1)
    logit = dot_exp / torch.clamp_min((dot_exp * (pmask + nmask)).sum(2, keepdim=True), 1e-5)
    loss = - torch.log(torch.clamp_min(logit, 1e-5)) * pmask
    loss_red = loss.sum() / torch.clamp_min(pmask.sum(), 1e-5)
    return loss_red
    '''

def _point_loss(logit, label):
    logit = torch_resize(logit, label)
    loss = F.cross_entropy(logit, label, ignore_index=255, reduction='sum')
    loss_red = loss / torch.clamp_min((label < 255).sum(), 1)
    return loss_red

def _point_loss_focal(logit, label):
    raise NotImplementedError

def _segmentation_loss(logit, label):
    loss = F.cross_entropy(logit, label, ignore_index=255)
    return loss

def _weighted_segmentation_loss(logit, label, use_weight_grad=False, add_mean=False):
    loss = F.cross_entropy(logit, label, ignore_index=255, reduction='none')
    if not use_weight_grad:
        logit = logit.detach()
    softmax = torch.softmax(logit, 1)
    weight = - (softmax * torch.log_softmax(logit, 1)).sum(1)
    if add_mean:
        weight = torch.clamp_min(weight, weight.mean())
    loss_red = (loss * weight).sum((1, 2)) / (weight.sum((1, 2)) + 1e-7)
    loss_red2 = loss_red.mean()
    return loss_red2, weight

@torch.no_grad()
def _pmask_from_logit(logit, image_label, threshold):
    softmax = torch.softmax(logit, 1)
    softmax = softmax * image_label.unsqueeze(2).unsqueeze(3).float()
    conf, pmask = softmax.max(1)
    if threshold:
        pmask[conf < threshold] = 255
    return pmask

@torch.no_grad()
def _pmask_from_logit_with_sp(logit, image_label, superpixel, threshold):
    softmax = torch.softmax(logit, 1)

    max_superpixel = superpixel.max()
    sp_oh = F.one_hot(superpixel, max_superpixel+1)[..., :max_superpixel].permute(0, 3, 1, 2).float()
    sp_size = torch.clamp_min(sp_oh.sum((2, 3)), 1e-3) # [N, M]

    avg_score = softmax.flatten(2) @ sp_oh.flatten(2).transpose(1, 2) #[N, C, M]
    avg_score = avg_score / sp_size.unsqueeze(1)
    avg_score = avg_score * image_label.unsqueeze(2)

    avg_score2d = (avg_score @ sp_oh.flatten(2)).view(logit.shape)
    conf, pmask = avg_score2d.max(1)
    if threshold:
        pmask[conf < threshold] = 255
    return pmask, (sp_oh, sp_size, avg_score)

## Methods to obtain reference features.
def _get_projections(proj_type, **kwargs):
    fn_src, fn_tgt = proj_type.split('2')
    is_self_proj = fn_src == fn_tgt
    fn_src = eval(f"_get_{fn_src}_proj")
    fn_tgt = eval(f"_get_{fn_tgt}_proj")
    src, src_lbl, _src_mask = fn_src(**kwargs)
    tgt, tgt_lbl, _tgt_mask = fn_tgt(**kwargs)
    if kwargs['cfg'].hyperparam.expand_batch:
        tgt = tgt.transpose(0, 1).contiguous().flatten(1).unsqueeze(0).expand(src.shape[0], -1, -1)
        tgt_lbl = tgt_lbl.flatten(0, 1).unsqueeze(0).expand(src.shape[0], -1, -1)
        is_self_proj = False
    _debug_masks = [_src_mask, _tgt_mask]
    return src, src_lbl, tgt, tgt_lbl, is_self_proj, _debug_masks

@torch.no_grad()
def _get_pos_neg_masks(src_lbl, tgt_lbl, is_self_proj, num_pos):
    '''
    src_lbl: [N, S, C]
    tgt_lbl: [N, T, C]
    pos, neg: [N, S, T]
    is_self_proj: bool
    num_pos: int
    '''
    pos = src_lbl @ tgt_lbl.transpose(1, 2) # [N, S, T]
    valid_query = pos.max(2)[0]
    neg = (1 - pos) * valid_query.unsqueeze(2)
    if is_self_proj:
        assert src_lbl.shape[1] == tgt_lbl.shape[1], (src_lbl.shape, tgt_lbl.shape)
        anti_eye = (1 - torch.eye(pos.shape[1], dtype=pos.dtype, device=pos.device)).unsqueeze(0).expand(pos.shape[0], -1, -1)
        pos = pos * anti_eye
    if num_pos:
        pos_priority = torch.rand(pos.shape, device=pos.device) * pos
        if num_pos == 1:
            pos_select = pos_priority.argmax(2)
            pos = F.one_hot(pos_select, pos.shape[2]).float() * pos
        else:
            pos_select = torch.topk(pos_priority, num_pos, dim=2)[1] # [N, S, num_pos]
            pos = F.one_hot(pos_select, pos.shape[2]).float().max(2)[0] * pos
    return pos, neg

def _get_pix_proj(proj, logit, pmask, **kwargs):
    proj = F.normalize(proj, dim=1).flatten(2) # [N, D, HW]
    #pmask = _pmask_from_logit(logit, image_label, 0.9)
    proj_lbl = F.one_hot(pmask.flatten(1), 256)[..., :logit.shape[1]].float() # [N, HW, C]
    return proj, proj_lbl, pmask

def _get_sp_proj(proj, sp_info, logit, cfg, **kwargs):
    proj = F.normalize(proj, dim=1) # [N, D, HW]
    sp_oh, sp_size, avg_score = sp_info

    sp_proj = proj.flatten(2) @ sp_oh.flatten(2).transpose(1, 2) # [N, D, M]
    sp_proj = sp_proj / sp_size.unsqueeze(1)
    sp_proj = F.normalize(sp_proj, dim=1)

    conf, sp_lbl = avg_score.max(1)
    sp_lbl[conf < cfg.hyperparam.threshold] = 255
    sp_lbl = F.one_hot(sp_lbl, 256)[..., :logit.shape[1]].float() # [N, M, C]
    return sp_proj, sp_lbl, None

def _get_img_proj(proj, logit, pmask, **kwargs):
    proj = F.normalize(proj, dim=1).flatten(2) # [N, D, HW]
    #pmask = _pmask_from_logit(logit, image_label, 0.9)
    proj_lbl = F.one_hot(pmask.flatten(1), 256)[..., :logit.shape[1]].float() # [N, HW, C]
    cnt_proj = torch.clamp_min(proj_lbl.sum(1), 1e-3) # [N, C]
    avg_proj = (proj @ proj_lbl) / cnt_proj.unsqueeze(1) # [N, D, C]
    avg_proj_lbl = torch.arange(logit.shape[1], device=logit.device, dtype=torch.int64).repeat(logit.shape[0], 1) # [N, C]
    avg_proj_lbl[cnt_proj < 5] = 255
    avg_proj_lbl = F.one_hot(avg_proj_lbl, 256)[..., :logit.shape[1]].float() # [N, C, C]
    return avg_proj, avg_proj_lbl, pmask

def _to_rank(data, rank):
    return [x.to(rank) for x in data]

## Main
def point_pmask_cl(rank, mod, data, niter=None, cfg=None):
    image, superpixel, label, image_label = _to_rank(data, rank)
    logit, proj = mod(image)
    if superpixel.shape[1:] != logit.shape[2:]:
        superpixel = torch_resize(superpixel, logit)

    # 1.
    loss_pnt = _point_loss(logit, label)

    # 2.
    if cfg.hyperparam.get('use_superpixel', True):
        pmask, sp_info = _pmask_from_logit_with_sp(logit, image_label, superpixel, cfg.hyperparam.threshold)
    else:
        pmask = _pmask_from_logit(logit, image_label, cfg.hyperparam.threshold)
    if cfg.hyperparam.weighted_map:
        loss_pmask, pmask_weight = _weighted_segmentation_loss(logit, pmask, cfg.hyperparam.use_weight_grad, cfg.hyperparam.add_mean)
    else:
        loss_pmask = _segmentation_loss(logit, pmask)

    # 3.
    src, src_lbl, tgt, tgt_lbl, is_self_proj, proj_masks = _get_projections(cfg.hyperparam.proj_type, **locals())

    if cfg.hyperparam.use_memory:
        if niter == 0:
            cfg._memer = TorchMemer(logit.shape[1], cfg.hyperparam.capacity)
            #cfg._memer = ClasswiseQueue(cfg.hyperparam.capacity, logit.shape[1])
        memer = cfg._memer
        with torch.no_grad():
            if cfg.hyperparam.expand_batch:
                tgt = tgt[0:1]
                tgt_lbl = tgt_lbl[0:1]
            tgt = tgt.transpose(1, 2).flatten(0, 1).contiguous() # [NT, D]
            tgt_lbl = tgt_lbl.flatten(0, 1) # [NT, C]
            memer.push(tgt.data, tgt_lbl.data)
            #tgt_lbl_exist, tgt_lbl = tgt_lbl.max(1)
            #tgt_lbl[tgt_lbl_exist < 1] = 255
            #memer.put(tgt.data, tgt_lbl.data)

            tgt, tgt_lbl = memer.pull()
            #tgt, tgt_lbl = memer.get()
            if tgt is None:
                # dummy tgt
                tgt = src.detach()[..., :cfg.hyperparam.num_pos].clone()
                tgt_lbl = torch.zeros_like(src_lbl[:, :cfg.hyperparam.num_pos, :])
            else:
                tgt = tgt.transpose(0, 1).unsqueeze(0).contiguous().expand(logit.shape[0], -1, -1)
                tgt_lbl = F.one_hot(tgt_lbl, logit.shape[1]).float().unsqueeze(0).expand(logit.shape[0], -1, -1)
            while tgt.shape[2] < cfg.hyperparam.num_pos:
                tgt = torch.cat([tgt, tgt], 2)
                tgt_lbl = torch.cat([tgt_lbl, tgt_lbl], 1)

    pos, neg = _get_pos_neg_masks(src_lbl, tgt_lbl, is_self_proj, cfg.hyperparam.num_pos)
    assert src.requires_grad, src.requires_grad
    assert src.shape[0] == logit.shape[0], (src.shape)

    focal = cfg.hyperparam.get('focal', False)
    if focal:
        loss_cl = _focal_contrastive_loss(src, tgt, pos, neg, power=cfg.hyperparam.focal_power)
    else:
        loss_cl = _contrastive_loss(src, tgt, pos, neg)

    # 0.
    if cfg.hyperparam.warmup:
        total_warmup = cfg.optimizer.niters_per_epoch * cfg.hyperparam.warmup
        warmup = min(niter, total_warmup) / total_warmup
    else:
        warmup = 1.0

    loss = loss_pnt + \
           loss_pmask * (cfg.hyperparam.lambda_pmask * warmup) +\
           loss_cl * (cfg.hyperparam.lambda_cl * warmup)

    monitor_vals = {
        "loss": loss,
        "loss_pnt": loss_pnt,
        "loss_pmask": loss_pmask,
        "loss_cl": loss_cl,
        "nPos": pos.sum(-1).mean(),
        "nNeg": neg.sum(-1).mean()
    }
    monitor_imgs = [image[0], label[0], pmask[0], logit[0]]
    for proj_mask in proj_masks:
        if proj_mask is not None:
            monitor_imgs.append(proj_mask[0])
    if cfg.hyperparam.weighted_map:
        monitor_imgs.append(pmask_weight[0])
    return loss,  monitor_vals, monitor_imgs