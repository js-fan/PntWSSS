from pathlib import Path
import sys
mmseg_path = Path(__file__).resolve().parents[1] / "trd_party/mmsegmentation"
sys.path.append(str(mmseg_path))

# from mmseg.models.backbones.resnet import ResNetV1c
# from mmseg.models.decode_heads.aspp_head import ASPPHead
# from mmseg.models.decode_heads.fcn_head import FCNHead
# from mmseg.models.segmentors.encoder_decoder import EncoderDecoder
from mmseg.models.builder import build_segmentor
from mmcv.utils import Config

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeeplabV3Plus(nn.Module):
    def __init__(self, cfg_file):
        super().__init__()
        cfg = Config.fromfile(cfg_file)
        assert cfg.model.type == "EncoderDecoder", \
                "Only EncoderDecoder model supported now."
        self.base = build_segmentor(cfg.model,
                train_cfg=cfg.get('train_cfg'),
                test_cfg=cfg.get('test_cfg'))
        self.base.init_weights()

    def forward(self, x, rtn_stages=None):
        feat = self.base.extract_feat(x) # [256s4, 512s8, 1024s8, 2048s8]
        logit = self.base.decode_head.forward(feat) # [cs4]
        
        # [cs8]
        if self.base.with_auxiliary_head:
            logit_aux = []
            if isinstance(self.base.auxiliary_head, nn.ModuleList):
                for idx, aux_head in enumerate(self.base.auxiliary_head):
                    logit_aux.append(aux_head.forward(feat))
            else:
                logit_aux.append(self.base.auxiliary_head.forward(feat))

        if self.training:
            return logit, logit_aux[0]
        else:
            return logit

    def get_param_groups(self):
        lr_mult_params = {1: list(self.parameters())}
        return lr_mult_params

    def init_params(self, pretrained, seed=None):
        pass

    '''
    def forward(self, x, rtn_stages=None):
        if rtn_stages is None:
            rtn_stages = [self.default_stage]
        data = self._forward_all(x)
        rtn_vals = [data[k] for k in _as_list(rtn_stages)]
        if len(rtn_vals) == 1:
            rtn_vals = rtn_vals[0]
        return rtn_vals


    def get_param_groups(self):
        lr_mult_params = {1: [], 10: list(self.fc8.parameters())}
    '''

if __name__ == '__main__':
    cfg_file = mmseg_path / 'configs/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_40k_cityscapes.py'
    assert cfg_file.is_file(), cfg_file
    mod = DeeplabV3Plus(str(cfg_file))
