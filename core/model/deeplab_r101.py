import torch.nn as nn
from .resnet import Bottleneck, ResNet

class _Resnet_DeepLab_Head(nn.Module):
    def __init__(self, num_classes, dilations=[6, 12, 18, 24]):
        super().__init__()
        inplanes = 2048

        layers = []
        for d in dilations:
            layers.append(nn.Conv2d(inplanes, num_classes, 3, 1, padding=d, dilation=d))
        self.heads = nn.ModuleList(layers)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        xs = [head(x) for head in self.heads]
        if len(xs) == 1:
            return xs[0]
        return sum(xs)

class DeeplabV2_R101(nn.Module):
    def __init__(self, num_classes, aspp=False):
        super().__init__()
        self.num_classes = num_classes
        self.aspp = aspp

        self.base = ResNet(Bottleneck, [3, 4, 23, 3], 8, nn.SyncBatchNorm, pretrained=True, use_mg=False)
        dilations = [6, 12, 18, 24] if aspp else [12]
        self.head = _Resnet_DeepLab_Head(num_classes, dilations)

    def init_params(self):
        pass

    def forward(self, image, rtn_dict=False):
        base, _low_level_feature = self.base(image)
        logit = self.head(base)
        if rtn_dict:
            return logit, {'base': base, 'logit': logit}
        return logit