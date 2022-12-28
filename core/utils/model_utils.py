import numpy as np
import torch
import warnings
from torch.optim.lr_scheduler import _LRScheduler
import torch.nn.functional as F
import cv2
from .image_utils import imhstack
from ..data.simple_point_dataset import SimplePointDataset

class Poly(_LRScheduler):
    def __init__(self, optimizer, num_epochs, power=0.9, last_epoch=-1, verbose=False):
        self.num_epochs = num_epochs
        self.power = power
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                                              "please use `get_last_lr()`.")

        if self.last_epoch <= 0:
            return list(self.base_lrs)
        decay = (1. - float(self.last_epoch) / self.num_epochs)**self.power
        lrs = [decay * lr for lr in self.base_lrs]
        return lrs

def get_params_lr_mult(mod, params_lr_mult):
    class _Lookup:
        def __init__(self, params_lr_mult, default_val):
            if params_lr_mult is None:
                self.lookup = {}
            else:
                self.lookup = dict(params_lr_mult)
            self.keys = list(self.lookup.keys())
            self.default_val = default_val

        def __contains__(self, key):
            for k in self.keys:
                if k in key:
                    return True
            return False

        def __getitem__(self, key):
            for k in self.keys:
                if k in key:
                    return self.lookup[k]
            return self.default_val

    out = {}
    lookup = _Lookup(params_lr_mult, 1)

    for k, v in mod.named_parameters():
        if not v.requires_grad:
            continue
        val = lookup[k]
        if val not in out:
            out[val] = [v]
        else:
            out[val].append(v)
    return out

def torch_resize(src, tgt, mode=None, align_corners=None):
    assert isinstance(src, torch.Tensor), type(src)
    assert src.ndim >= 2 and src.ndim <= 4, src.shape

    expand = 4 - src.ndim
    for _ in range(expand):
        src = src.unsqueeze(0)

    if mode is None:
        mode = {
                torch.float32: 'bilinear',
                torch.int64:   'nearest',
        }[src.dtype]

    if isinstance(tgt, torch.Tensor):
        tgt = tgt.shape[-2:]
    if isinstance(tgt, (list, tuple)):
        assert len(tgt) == 2, tgt
        out = F.interpolate(src.float(), size=tgt, mode=mode, align_corners=align_corners)
    elif isinstance(tgt, (int, float)):
        out = F.interpolate(src.float(), scale_factor=tgt, mode=mode, align_corners=align_corners)
    else:
        raise RuntimeError(tgt)

    out = out.to(src.dtype)
    for _ in range(expand):
        out = out.squeeze(0)
    return out

class SEGVisualizer(object):
    def __init__(self, num_classes, palette, ignore_class=255):
        self.num_classes = num_classes
        self.palette = palette
        self.sp_palette = np.random.randint(0, 256, (2048, 3), np.uint8)
        self.ignore_class = ignore_class

    def get_img(self, x):
        return SimplePointDataset.denormalize(x)

    def get_seg(self, x):
        x = self.palette[x.astype(np.int64).ravel()].reshape(x.shape+(3,))
        return x

    def get_sp(self, x):
        x = self.sp_palette[x.astype(np.int64).ravel() % 2048].reshape(x.shape+(3,))
        return x

    def get_heatmap(self, x, img=None):
        if x.ndim == 2:
            x = x[np.newaxis]
        x = x - x.min((1, 2), keepdims=True)
        x = x / np.maximum(x.max((1, 2), keepdims=True), 1e-5)
        x = (x.max(0) * 255).astype(np.uint8)
        x = cv2.applyColorMap(x, cv2.COLORMAP_JET)
        if img is not None:
            x = cv2.addWeighted(img, 0.2, cv2.resize(x, img.shape[:2][::-1]), 0.8, 0)
        return x

    def visualize(self, *data, heatmap=False, stack='h', size=360):
        assert stack in ['h', 'v']
        outs = []
        for x in data:
            assert x.ndim >= 2 and x.ndim <= 3, [x.shape for x in data]
            if isinstance(x, torch.Tensor):
                x = x.data.cpu().numpy()
            assert isinstance(x, np.ndarray), type(x)

            if x.ndim == 3:
                if x.shape[0] == 3:
                    outs.append(self.get_img(x))
                elif x.shape[0] == self.num_classes:
                    if heatmap:
                        outs.append(self.get_heatmap(x, outs[0]))
                    outs.append(self.get_seg(x.argmax(0)))
                else:
                    outs.append(self.get_sp(x.argmax(0)))
            else:
                if (x.astype(np.int32).astype(x.dtype) != x).sum() > 0:
                    outs.append(self.get_heatmap(x))
                else:
                    unique_vals = [v for v in np.unique(x) if v != self.ignore_class]
                    if (len(unique_vals) == 0) or (np.array(unique_vals).max() < self.num_classes):
                        outs.append(self.get_seg(x))
                    else:
                        outs.append(self.get_sp(x))
        return eval(f"im{stack}stack")(outs, size)

    def __call__(self, *args, **kwargs):
        return self.visualize(*args, **kwargs)

