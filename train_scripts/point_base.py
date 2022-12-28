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

def point_base(rank, mod, data, niter=None, cfg=None):
    image, label = data
    logit = mod(image.to(rank))

    loss = _point_loss(logit, label.to(rank))

    monitor_vals = {
        "loss": loss,
    }
    monitor_imgs = [image[0], label[0], logit[0]]
    return loss,  monitor_vals, monitor_imgs