import torch
import torch.nn as nn
import torch.nn.functional as F
from core.utils import *

def _cross_entropy_2d(logit, label):
    loss = F.cross_entropy(logit, label, ignore_index=255, reduction='sum')
    loss = loss / torch.clamp_min((label < 255).sum(), 1)
    return loss

def point_with_superpixel(rank, mod, data, niter, cfg):
    image, superpixel, label = data
    logit = mod(image.to(rank))
    label = torch_resize(label.to(rank), logit)
    loss_pnt = _cross_entropy_2d(logit, label)

    with torch.no_grad():
        sp_mask = F.one_hot(superpixel.to(rank), cfg.data.train.max_superpixel+1)
        sp_mask = sp_mask[..., :cfg.data.train.max_superpixel].permute(0,3,1,2)
        sp_mask = torch_resize(sp_mask.float(), logit).contiguous() # [N, M, H, W]
        sp_logit = logit.flatten(2) @ sp_mask.flatten(2).transpose(1,2) # [N, C, M]
        sp_logit = sp_logit / torch.clamp_min(sp_mask.sum((2, 3)).unsqueeze(1), 1e-5)

        sp_prob = torch.softmax(sp_logit, 1) # [N, C, M]
        sp_prob2d = (sp_prob @ sp_mask.flatten(2)).view(logit.shape) 

        sp_conf, sp_lbl = sp_prob2d.max(1)
        if cfg.hyperparam.get('use_entropy', False):
            sp_entropy = (-sp_prob2d * torch.log(sp_prob2d+1e-5)).sum(1)
            sp_conf = 1 - sp_entropy / np.log(logit.shape[1])
        sp_lbl[sp_conf < cfg.hyperparam.superpixel_threshold] = 255

    loss_sp = _cross_entropy_2d(logit, sp_lbl)

    warmup = min(float(niter) / (cfg.hyperparam.warmup * cfg.optimizer.niters_per_epoch), 1)
    loss = loss_pnt + loss_sp * (cfg.hyperparam.lambda_superpixel * warmup)

    return loss, {'loss_pnt': loss_pnt, 'loss_sp': loss_sp}, [image[0], label[0], sp_lbl[0], logit[0]]

