import torch
import torch.nn as nn
import torch.nn.functional as F
from .vgg_base import VGG16

__all__ = [
        "ProjSegModel",
        "ProjSegModelUseSP"
]

class MSProjModel(nn.Module):
    def __init__(self,
            proj_in_channels,
            proj_out_channels,
            proj_mid_channels=None,
            normalize=True
        ):
        super().__init__()
        self.proj_in_channels = proj_in_channels
        self.proj_out_channels = proj_out_channels
        self.proj_mid_channels = proj_mid_channels if proj_mid_channels is not None \
                else proj_out_channels
        self.normalize = normalize

        self.linear_a = nn.ModuleList([
            nn.Linear(chin, self.proj_mid_channels, bias=False) \
                    for chin in self.proj_in_channels])
        self.linear_b = nn.Linear(self.proj_mid_channels*len(self.proj_in_channels),
                self.proj_out_channels, bias=False)
        self._reset_params()

    def _reset_params(self):
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.normal_(mod.weight, 0, 0.01)

    def forward(self, x):
        assert len(x) == len(self.proj_in_channels), (len(x), len(self.proj_in_channels))
        proj_a = [F.relu(linear_a(x_i)) for linear_a, x_i in zip(self.linear_a, x)]
        proj_a = torch.cat(proj_a, -1)
        proj_b = self.linear_b(proj_a)
        if self.normalize:
            proj_b = F.normalize(proj_b, dim=-1)
        return proj_b

    def get_param_groups(self):
        lr_mult_params = {10: list(self.parameters())}
        return lr_mult_params

class ProjSegModel(nn.Module):
    def __init__(self,
            base,
            base_kwargs,
            proj_stages,
            proj_in_channels,
            proj_out_channels,
            proj_mid_channels=None,
            normalize=True,
            proj_size=None
            ):
        super().__init__()
        assert isinstance(proj_stages, (list, tuple)) and len(proj_stages) >= 1
        assert len(proj_in_channels) == len(proj_stages)

        self.base = eval(base)(**base_kwargs)
        self.base_stages = [self.base.default_stage] + list(proj_stages)
        self.proj_mod = MSProjModel(proj_in_channels, proj_out_channels, proj_mid_channels, normalize)
        self.normalize = normalize
        self.proj_size = proj_size
        self.proj_out_channels = proj_out_channels

    def _resize_feats(self, feats, size):
        return [F.interpolate(feat, size, mode='bilinear', align_corners=True) for feat in feats]

    def forward(self, x):
        seg, *feats = self.base(x, self.base_stages)
        if not self.training:
            return seg

        proj_size = feats[-1].shape[-2:] if self.proj_size is None else self.proj_size
        feats = self._resize_feats(feats, proj_size)
        feats = [feat.flatten(2).transpose(1, 2) for feat in feats]
        if self.normalize:
            feats = [F.normalize(feat, dim=-1) for feat in feats]
        proj = self.proj_mod(feats)
        proj = proj.transpose(1, 2).view(seg.shape[0], self.proj_out_channels, proj_size[0], proj_size[1])
        return seg, proj

    def init_params(self, pretrained, seed=None):
        self.base.init_params(pretrained, seed)

    def get_param_groups(self):
        base_params = self.base.get_param_groups()
        proj_params = self.proj_mod.get_param_groups()
        lr_mult_params = {k: base_params.get(k, []) + proj_params.get(k, []) \
                for k in set(list(base_params.keys()) + list(proj_params.keys()))}
        return lr_mult_params

class ProjSegModelUseSP(ProjSegModel):
    def __init__(self,
            base,
            base_kwargs,
            proj_stages,
            proj_in_channels,
            proj_out_channels,
            proj_mid_channels=None,
            normalize=True,
            proj_size=None
            ):
        super().__init__(
                base,
                base_kwargs,
                proj_stages,
                proj_in_channels,
                proj_out_channels,
                proj_mid_channels,
                normalize,
                proj_size)

    def _avg_feats(self, feats, sp_mask):
        sp_flat = sp_mask.flatten(2)
        sp_sum = torch.clamp_min(sp_flat.sum(2, keepdim=True), 1)
        outs = [(sp_flat @ feat.flatten(2).transpose(1, 2)) / sp_sum for feat in feats]
        return outs

    def forward(self, x, sp_mask=None, map2d=False, rtn_feats=False):
        seg, *feats = self.base(x, self.base_stages)
        if rtn_feats:
            return seg, feats

        if not self.training:
            return seg

        proj_size = sp_mask.shape[-2:]
        feats = self._resize_feats(feats, proj_size)
        feats = self._avg_feats(feats, sp_mask)
        if self.normalize:
            feats = [F.normalize(feat, dim=-1) for feat in feats]
        proj = self.proj_mod(feats).transpose(1, 2) # NCM
        if map2d:
            proj = (proj @ sp_mask.flatten(2)).view(seg.shape[0], self.proj_out_channels, proj_size[0], proj_size[1])
        return seg, proj

