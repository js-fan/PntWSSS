import torch
import torch.nn as nn
import torch.nn.functional as F
#from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from .aspp import build_aspp
from .decoder import build_decoder
from .resnet import ResNet101

class DeepLab_V3p(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=False, freeze_bn=False, embedding='', seed=None):
        assert seed is not None, 'seed is required'
        torch.manual_seed(seed)
        assert backbone == 'resnet', 'Other backbones have not been implemented: {}'.format(backbone)

        super(DeepLab_V3p, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        # sync_bn should be implemented by torch.nn.SyncBatchNorm.convert_sync_batchnorm(model) before dds.
        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        #self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.backbone = ResNet101(output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)

        self.freeze_bn = freeze_bn

        # additional embedding for metric learning
        if embedding:
            self._embedding_type = embedding
            self._in_embed_channels = 2048
            if embedding == 'linear':
                self.embedding = nn.Conv2d(self._in_embed_channels, 256, 1, bias=False)
            elif embedding == 'mlp':
                self.embedding = nn.Sequential(
                        nn.Conv2d(self._in_embed_channels, 512, 1, bias=False),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(512, 256, 1, bias=False)
                )
            else:
                raise RuntimeError(embedding)
            self._init_embedding()
        else:
            self.embedding = None

    def _init_embedding(self):
        for m in self.embedding.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)

    def set_mode(self, mode):
        self._forward_mode = mode

    def forward(self, input):
        return getattr(self, 'forward_' + self._forward_mode)(input)

    def forward_seg(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        return x

    def forward_seg_embed(self, input):
        assert self.embedding is not None
        x, low_level_feat = self.backbone(input)
        embed = self.embedding(x)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        return x, embed

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_param_groups(self):
        lr_1x = self.get_1x_lr_params()
        lr_10x = self.get_10x_lr_params()
        return {1: lr_1x, 10: lr_10x}


if __name__ == "__main__":
    model = DeepLab_V3p(backbone='mobilenet', output_stride=16)
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output.size())


