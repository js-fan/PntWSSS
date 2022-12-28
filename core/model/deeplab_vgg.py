import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import warnings
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

class DeeplabV2_VGG16(nn.Module):
    def __init__(self, num_classes, pretrained=None, drop=0.5, dilation=12):
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
        self.pretrained = pretrained
        self.init_params()

    def init_params(self):
        if self.pretrained:
            assert os.path.exists(self.pretrained), self.pretrained
            pretrained = torch.load(self.pretrained, 'cpu')
            if list(pretrained.keys())[0].startswith("module."):
                pretrained = {k.replace("module.", ""): v for k, v in pretrained.items()}
            if "fc8.weight" in pretrained:
                del pretrained["fc8.weight"], pretrained["fc8.bias"]
            pretrained["fc8.weight"] = torch.normal(0, 0.01, (self.num_classes,1024,1,1), dtype=torch.float32)
            pretrained["fc8.bias"] = torch.zeros((self.num_classes,), dtype=torch.float32)
            self.load_state_dict(pretrained, strict=True)
        else:
            warnings.warn("No pretrained parameters found!!!")
            for mod in self.modules():
                if isinstance(mod, nn.Conv2d):
                    nn.init.xavier_normal_(mod.weight)
                    if mod.bias is not None:
                        nn.init.zeros_(mod.bias)

    def forward(self, image, rtn_dict=False):
        c1 = self.conv1(image)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)
        c5 = self.avg_pool(c5)
        c6 = self.fc6(c5)
        c7 = self.fc7(c6)
        c8 = self.fc8(c7)
        if rtn_dict:
            return c8, {k: v for k, v in locals().items() if re.match(r"c\d+$", k)}
        return c8

