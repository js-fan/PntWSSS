import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import warnings

from .deeplab_vgg import *
from .deeplab_r101 import *

class ProjectionModel(nn.Module):
    def __init__(self, base, base_args,
            base_stage_names,
            base_stage_dims,
            proj_mid_dims,
            proj_out_dims,
            sync_scale=-1,
            normalize=True
            ):
        super().__init__()
        assert len(base_stage_names) == len(base_stage_dims), (base_stage_names, base_stage_dims)
        self.base = eval(base)(**base_args)
        self.base_stage_names = base_stage_names
        self.base_stage_dims = base_stage_dims
        self.proj_mid_dims = proj_mid_dims if proj_mid_dims is not None else proj_out_dims
        self.proj_out_dims = proj_out_dims
        self.sync_scale = sync_scale
        self.normalize = normalize

        self.proj_mid = nn.ModuleList([nn.Conv2d(dim_in, self.proj_mid_dims, 1, bias=False) \
                    for dim_in in self.base_stage_dims])
        self.proj_out = nn.Conv2d(self.proj_mid_dims*len(base_stage_dims), self.proj_out_dims, 1, bias=False)

    def forward(self, image):
        logit, stage_dict = self.base(image, True)
        if not self.training:
            return logit

        feats = [stage_dict[name] for name in self.base_stage_names]

        sync_size = feats[self.sync_scale].shape[-2:]
        for i in range(len(feats)):
            feat = feats[i]
            if feat.shape[:-2] != sync_size:
                feats[i] = F.interpolate(feat, sync_size, mode='bilinear', align_corners=True)

        if self.normalize:
            feats = [F.normalize(feat, dim=1) for feat in feats]
        feats = [F.relu(proj_fn(feat), inplace=True) for feat, proj_fn in zip(feats, self.proj_mid)]
        feats = torch.cat(feats, 1)
        proj = self.proj_out(feats)

        return logit, proj
