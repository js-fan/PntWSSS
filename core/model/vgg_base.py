import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import re

def _make_conv_block(num_conv, num_filter_in, num_filter_out, pool=None, dilation=1, inplace=True, drop=None, kernel=3):
    pad = (kernel // 2) if (dilation == 1) else dilation
    layers = OrderedDict()

    for i in range(num_conv):
        _num_filter_in = num_filter_in if i == 0 else num_filter_out
        layers['{}'.format(i)] = nn.Conv2d(_num_filter_in, num_filter_out, kernel, 1, pad, dilation, bias=True)
        layers['{}_relu'.format(i)] = nn.ReLU(inplace=inplace)
    if pool:
        layers['pool'] = nn.MaxPool2d(3, pool, 1)
    if drop:
        layers['drop'] = nn.Dropout(drop)
    return nn.Sequential(layers)

def _as_list(x):
    return [x] if not isinstance(x, (list, tuple)) else list(x)

class VGG16(nn.Module):
    def __init__(self, num_classes, dilation=12, drop=0.5):
        super().__init__()
        self.conv1 = _make_conv_block(2,   3,  64, 2)
        self.conv2 = _make_conv_block(2,  64, 128, 2)
        self.conv3 = _make_conv_block(3, 128, 256, 2)
        self.conv4 = _make_conv_block(3, 256, 512, 1)
        self.conv5 = _make_conv_block(3, 512, 512, 1, 2)
        self.avg_pool = nn.AvgPool2d(3, 1, 1)
        self.fc6 = _make_conv_block(1, 512, 1024, drop=drop, dilation=dilation)
        self.fc7 = _make_conv_block(1, 1024, 1024, drop=drop, kernel=1)
        self.fc8 = nn.Conv2d(1024, num_classes, 1)

        self.num_classes = num_classes
        self.default_stage = "c8"

    def _forward_all(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)
        c5 = self.avg_pool(c5)
        c6 = self.fc6(c5)
        c7 = self.fc7(c6)
        c8 = self.fc8(c7)
        data = {k: v for k, v in locals().items() if re.match(r"c\d+$", k)}
        return data

    def forward(self, x, rtn_stages=None):
        if rtn_stages is None:
            rtn_stages = [self.default_stage]
        data = self._forward_all(x)
        rtn_vals = [data[k] for k in _as_list(rtn_stages)]
        if len(rtn_vals) == 1:
            rtn_vals = rtn_vals[0]
        return rtn_vals

    def init_params(self, pretrained, seed=None):
        if isinstance(pretrained, str):
            pretrained = torch.load(pretrained, map_location="cpu")

        if list(pretrained.keys())[0].startswith("module."):
            pretrained = {k.replace("module.", ''): v for k, v in pretrained.items()}

        if seed is not None:
            torch.manual_seed(seed)

        has_fc8_weights = ("fc8.weight" in pretrained) and \
                (pretrained["fc8.weight"].shape == (self.num_classes, 1024, 1, 1))
        if not has_fc8_weights:
            pretrained["fc8.weight"] = torch.normal(0, 0.01, (self.num_classes, 1024, 1, 1),
                    dtype=torch.float32)
            pretrained["fc8.bias"] = torch.zeros((self.num_classes,), dtype=torch.float32)

        self.load_state_dict(pretrained, strict=True)

    def get_param_groups(self):
        lr_mult_params = {1: [], 10: list(self.fc8.parameters())}
        isParamIn = lambda x, plist: any([y is x for y in plist])
        for param in self.parameters():
            if param.requires_grad and not isParamIn(param, lr_mult_params[10]):
                lr_mult_params[1].append(param)
        return lr_mult_params

