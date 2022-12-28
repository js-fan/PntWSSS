import torch
import torch.nn as nn
import torch.nn.functional as F
from core.utils import *

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
    #softmax = softmax * image_label.unsqueeze(2).unsqueeze(3).float()

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
    return pmask

def point_pmask(rank, mod, data, niter=None, cfg=None):
    image, superpixel, label, image_label = data
    logit = mod(image.to(rank))

    loss_pnt = _point_loss(logit, label.to(rank))

    if cfg.hyperparam.use_superpixel:
        pmask = _pmask_from_logit_with_sp(logit, image_label.to(rank), superpixel.to(rank), cfg.hyperparam.threshold)
    else:
        pmask = _pmask_from_logit(logit, image_label.to(rank), cfg.hyperparam.threshold)
    loss_pmask = _segmentation_loss(logit, pmask)

    if cfg.hyperparam.warmup:
        total_warmup = cfg.optimizer.niters_per_epoch * cfg.hyperparam.warmup
        warmup = min(niter, total_warmup) / total_warmup
    else:
        warmup = 1.0

    loss = loss_pnt + loss_pmask * warmup

    monitor_vals = {
        "loss": loss,
        "loss_pnt": loss_pnt,
        "loss_pmask": loss_pmask,
        "warmup": warmup
    }
    monitor_imgs = [image[0], label[0], pmask[0], logit[0]]
    return loss,  monitor_vals, monitor_imgs