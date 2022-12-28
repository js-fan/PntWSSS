import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

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

class vgg16_largefov(nn.Module):
    def __init__(self, num_classes, dilation=12, upsample=1, drop=0.5,
            embedding='', concat_layers = ['conv5', 'fc6', 'fc7'], mode='seg'):
        super(vgg16_largefov, self).__init__()
        self.conv1 = _make_conv_block(2,   3,  64, 2)
        self.conv2 = _make_conv_block(2,  64, 128, 2)
        self.conv3 = _make_conv_block(3, 128, 256, 2)
        self.conv4 = _make_conv_block(3, 256, 512, 1)
        self.conv5 = _make_conv_block(3, 512, 512, 1, 2)
        self.avg_pool = nn.AvgPool2d(3, 1, 1)
        self.fc6 = _make_conv_block(1, 512, 1024, drop=drop, dilation=dilation)
        self.fc7 = _make_conv_block(1, 1024, 1024, drop=drop, kernel=1)
        self.fc8 = nn.Conv2d(1024, num_classes, 1)

        self._num_classes = num_classes
        self._forward_mode = None
        self._upsample = upsample

        # additional embedding on cat('conv5', 'fc6', 'fc7')
        if embedding:
            layer_channels = {'conv3': 256, 'conv4': 512, 'conv5': 512, 'fc6': 1024, 'fc7': 1024}
            self._in_embed_channels = sum([layer_channels[layer] for layer in concat_layers])
            if embedding == 'linear':
                self.embedding = nn.Conv2d(self._in_embed_channels, 256, 1, bias=False)
            elif embedding == 'mlp':
                self.embedding = nn.Sequential(
                        nn.Conv2d(self._in_embed_channels, 512, 1, bias=False),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(512, 256, 1, bias=False)
                )
            else:
                raise ValueError(embedding)
            self._embedding_type = embedding
            self._concat_layers = concat_layers
        else:
            self.embedding = None

        self.set_mode(mode)

    def set_mode(self, mode):
        for item in mode.lower().split('_'):
            assert hasattr(self, '_get_' + item), "Error: unknown forward item '{}'.".format(item)
        self._forward_mode = mode.lower()

    def _upsample_if_need(self, x, upsample=None):
        upsample = self._upsample if upsample is None else upsample
        if upsample == 1:
            return x

        h, w = x.size()[2:]
        new_h = int(upsample * (h - 1) + 1)
        new_w = int(upsample * (w - 1) + 1)
        x_out = F.interpolate(x, (new_h, new_w), mode='bilinear')
        return x_out

    def _get_cls(self, base):
        return base['fc8'].mean(axis=(2, 3))

    def _get_seg(self, base):
        return self._upsample_if_need(base['fc8'])

    def _get_cam(self, base, upsample=True):
        with torch.no_grad():
            cam = torch.clamp_min(base['fc8'], 1e-5)
            cam = cam / cam.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
            if upsample:
                cam = self._upsample_if_need(cam)
        return cam

    def _get_cambg(self, base, upsample=True):
        with torch.no_grad():
            cam = torch.clamp_min(base['fc8'], 1e-5)
            cam = cam / cam.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
            cam_sigmoid = torch.sigmoid(base['fc8'])

            label = base['label']
            if label is not None:
                cam = cam * label.view(label.size() + (1, 1))
                cam_sigmoid = cam_sigmoid * label.view(label.size() + (1, 1))

            bg = 1 - cam_sigmoid.max(dim=1, keepdim=True)[0]
            cambg = torch.cat([bg, cam], dim=1)
            if upsample:
                cambg = self._upsample_if_need(cambg)
        return cambg

    def _get_mscambg(self, base, upsample=True):
        x_origin = base['x']
        label = base['label']
        cand_cambg = []

        with torch.no_grad():
            cand_cambg.append(self._get_cambg(base, False))
            origin_size = cand_cambg[0].size()[2:]
            for scale in [0.75, 1.25]:
                x = F.interpolate(x_origin, scale_factor=scale, mode='bilinear', align_corners=True)
                _base = self._get_base(x, label, False, True)
                cambg = F.interpolate(self._get_cambg(_base, False), size=origin_size, mode='bilinear', align_corners=True)
                cand_cambg.append(cambg)
            
            cambg = sum(cand_cambg) / len(cand_cambg)
            if upsample:
                cambg = self._upsample_if_need(cambg)
        return cambg

    def _get_conv2(self, base):
        return base['conv2']

    def _get_conv3(self, base):
        return base['conv3']

    def _get_conv4(self, base):
        return base['conv4']

    def _get_conv5(self, base):
        return base['conv5']

    def _get_fc6(self, base):
        return base['fc6']

    def _get_fc7(self, base):
        return base['fc7']

    def _get_embed(self, base):
        feature_list = [base[layer] for layer in self._concat_layers]
        feature = torch.cat(feature_list, dim=1)
        #feature = torch.cat([base['conv5'], base['fc6'], base['fc7']], dim=1)
        out = self.embedding(feature)
        out = self._upsample_if_need(out)
        return out

    def _get_base(self, x, label=None):
        base = OrderedDict()
        base['x'] = x
        base['label'] = label
        base['conv1'] = self.conv1(x)
        base['conv2'] = self.conv2(base['conv1'])
        base['conv3'] = self.conv3(base['conv2'])
        base['conv4'] = self.conv4(base['conv3'])
        base['conv5'] = self.avg_pool(self.conv5(base['conv4']))

        base['fc6'] = self.fc6(base['conv5'])
        base['fc7'] = self.fc7(base['fc6'])
        base['fc8'] = self.fc8(base['fc7'])

        return base

    def forward(self, x, label=None):
        base = self._get_base(x, label)
        out_vars = []
        for item in self._forward_mode.split('_'):
            out_vars.append( getattr(self, '_get_' + item)(base) )
        if len(out_vars) == 1:
            return out_vars[0]
        return tuple(out_vars)

    def convert_from_mxnet(self, params: str) -> OrderedDict:
        import mxnet as mx
        mx_params = mx.nd.load(params)
        torch_params = OrderedDict()
        for k, _ in self.named_parameters():
            if k.startswith('conv'):
                block, layer, suffix = k.split('.')
                name = 'arg:{}_{}_{}'.format(block, int(layer)+1, suffix)
            elif k.startswith('fc'):
                name_split = k.split('.')
                block, suffix = name_split[0], name_split[-1]
                name = 'arg:{}_{}'.format(block, suffix)
            else:
                raise ValueError(k)
            print(k, name)
            torch_params[k] = torch.from_numpy(mx_params[name].asnumpy())
        return torch_params

    def init_params(self, pretrained: OrderedDict, seed: int) -> None:
        # unwarp if necessary, in cases pretrained model is warped by nn.Parallel
        pretrained_ = OrderedDict()
        for k, v in pretrained.items():
            if k.startswith('module.'):
                k = k.split('.', 1)[1]
            pretrained_[k] = v
        pretrained = pretrained_

        torch.manual_seed(seed)

        # init fc8
        init_fc8 = True
        if 'fc8.weight' in pretrained:
            prev_fc8_size = pretrained['fc8.weight'].size()
            curr_fc8_size = (self._num_classes, 1024, 1, 1)
            if prev_fc8_size == curr_fc8_size:
                init_fc8 = False

        if init_fc8:
            pretrained['fc8.weight'] = torch.zeros((self._num_classes, 1024, 1, 1), dtype=torch.float32)
            pretrained['fc8.bias'] = torch.zeros((self._num_classes,), dtype=torch.float32)
            nn.init.normal_(pretrained['fc8.weight'], 0, 0.01)

        # init embedding layers
        if self.embedding is not None:
            if self._embedding_type == 'linear':
                pretrained['embedding.weight'] = torch.zeros((256, self._in_embed_channels, 1, 1), dtype=torch.float32)
                nn.init.normal_(pretrained['embedding.weight'], 0, 0.01)
            elif self._embedding_type == 'mlp':
                pretrained['embedding.0.weight'] = torch.zeros((512, self._in_embed_channels, 1, 1), dtype=torch.float32)
                pretrained['embedding.2.weight'] = torch.zeros((256, 512, 1, 1), dtype=torch.float32)
                nn.init.normal_(pretrained['embedding.0.weight'], 0, 0.01)
                nn.init.normal_(pretrained['embedding.2.weight'], 0, 0.01)

        self.load_state_dict(pretrained, strict=True)

    def get_param_groups(self):
        lr_mult_params = OrderedDict()
        lr_mult_params[1] = []
        lr_mult_params[10] = []

        for k, v in self.named_parameters():
            if not v.requires_grad:
                continue
            if k in ['fc8.weight', 'fc8.bias']:
                lr_mult_params[10].append(v)
                #elif k.startswith('embedding.'):
                #lr_mult_params[10].append(v)
            else:
                lr_mult_params[1].append(v)

        # post-check empty param_groups:
        empty_param_groups = []
        for k, v in lr_mult_params.items():
            if len(v) == 0:
                empty_param_groups.append(k)
        for k in empty_param_groups:
            del lr_mult_params[k]

        return lr_mult_params

