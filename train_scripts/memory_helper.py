import torch
import torch.nn as nn
import torch.nn.functional as F
from core.utils import *

class CircularTensor(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.data = None

    def push(self, data):
        assert data.ndim == 2, data.shape
        if self.data is None:
            self.data = data[:self.capacity]
        elif data.shape[0] >= self.capacity:
            self.data = data[:self.capacity]
        else:
            len_after_cat = self.data.shape[0] + data.shape[0]
            len_need_rm = len_after_cat - self.capacity
            if len_need_rm > 0:
                self.data = self.data[len_need_rm:]
            self.data = torch.cat([self.data, data])

    def pull(self, clone=False):
        if self.data is None:
            return None

        if clone:
            return self.data.clone()
        return self.data

    def __len__(self):
        return 0 if self.data is None else len(self.data)

class TorchMemer(object):
    def __init__(self, num_classes, capacity):
        self.num_classes = num_classes
        self.capacity = capacity
        self.data = [CircularTensor(capacity) for _ in range(num_classes)]

    @torch.no_grad()
    def push(self, data, label):
        assert data.ndim == 2, data.shape # [N, D]
        assert label.ndim <= 2, label.shape # [N,] or [N, C]
        if label.ndim == 1:
            label = F.one_hot(label, 256)[..., :self.num_classes]
        label = label.to(bool)
        for i in range(self.num_classes):
            mask = label[:, i]
            data_c = data[mask]
            if data_c.shape[0] > 0:
                self.data[i].push(data_c)
    
    @torch.no_grad()
    def pull_element_with_label(self, i, clone=False):
        data = self.data[i].pull(clone)
        if data is None:
            return None
        label = torch.full((data.shape[0],), i, dtype=torch.int64, device=data.device)
        return data, label

    @torch.no_grad()
    def pull(self, label=None, clone=False):
        # [N, D], [N,]
        if label is None:
            out_dl = [self.pull_element_with_label(i, clone) for i in range(self.num_classes)]
        elif label.ndim == 1:
            label = torch.unique(label)
            out_dl = [self.pull_element_with_label(i, clone) for i in label]
        elif label.ndim == 2:
            assert label.shape[1] == self.num_classes
            label_global = torch.nonzero(label.max(0)[0], as_tuple=True)[0]
            out_dl = [self.pull_element_with_label(i, clone) for i in label_global]
        else:
            raise RuntimeError(label.shape)

        data = [x[0] for x in out_dl if x is not None]
        label = [x[1] for x in out_dl if x is not None]
        if len(data) == 0:
            return None, None
        return torch.cat(data), torch.cat(label)
