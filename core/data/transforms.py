from os import supports_bytes_environ
import torch
import numpy as np
import cv2

_norm_mean = (0.406, 0.456, 0.485)
_norm_std = (0.225, 0.224, 0.229)

class TransformableData(object):
    @classmethod
    def load(cls, src):
        return cls(cv2.imread(str(src)))

    def __init__(self, data):
        self.data = data

    def transpose(self, *args):
        self.data = self.data.transpose(*args)

    def get(self):
        return self.data
    
    def to_torch(self):
        data = self.data.copy()
        if isinstance(self, (TransformableImage, TransformablePoint)):
            data = data.astype(np.float32)
        else:
            data = data.astype(np.int64)
        return torch.from_numpy(data)

    @property
    def shape(self):
        return self.data.shape

    def scale(self, s, mode=cv2.INTER_LINEAR):
        h, w = self.data.shape[:2]
        h2, w2 = int(h * s + .5), int(w * s + .5)
        self.data = cv2.resize(self.data, (w2, h2), interpolation=mode)

    def resize(self, size, mode=cv2.INTER_LINEAR):
        self.data = cv2.resize(self.data, tuple(size), interpolation=mode)

    def mirror(self):
        self.data = self.data[:, ::-1]

    def pad(self, top, bottom, left, right, value=(104, 117, 124)):
        self.data = cv2.copyMakeBorder(self.data, top, bottom, left, right, cv2.BORDER_CONSTANT, value=value)

    def crop(self, top, bottom, left, right):
        self.data = self.data[top:bottom, left:right]

    def normalize(self):
        self.data = ((self.data.astype(float)/255 - _norm_mean) / _norm_std)[..., ::-1].astype(np.float32)

    def denormalize(self):
        self.data = (((self.data[..., ::-1] * _norm_std) + _norm_mean) * 255).astype(np.uint8)
        return self

class TransformableImage(TransformableData):
    def __init__(self, data):
        assert isinstance(data, np.ndarray), type(data)
        assert data.shape[2] == 3, data.shape
        super().__init__(data)
    
class TransformableLabel2D(TransformableData):
    @classmethod
    def load(cls, src):
        return cls(cv2.imread(str(src), 0))

    def __init__(self, data, ignore_class=255):
        assert isinstance(data, np.ndarray), type(data)
        assert data.ndim == 2, data.shape
        super().__init__(data)
        self.ignore_class = ignore_class

    def scale(self, s):
        super().scale(s, cv2.INTER_NEAREST)

    def resize(self, size):
        super().resize(size, cv2.INTER_NEAREST)

    def pad(self, top, bottom, left, right, value=None):
        value = self.ignore_class if value is None else value
        super().pad(top, bottom, left, right, value)

    def normalize(self):
        pass
    def denormalize(self):
        pass

class TransformableSuperpixel(TransformableLabel2D):
    @classmethod
    def load(cls, src):
        return cls(cv2.imread(str(src)))

    def __init__(self, data):
        assert isinstance(data, np.ndarray), type(data)
        assert data.ndim == 3, data.shape
        data = data.astype(np.int64)
        data = data[..., 0] + data[..., 1] * 256 + data[..., 2] * 65536
        super().__init__(data)

    def pad(self, top, bottom, left, right):
        value = int(self.data.max()) + 1
        super().pad(top, bottom, left, right, value)

    def rearrange(self, max_num=None, min_size=0):
        sp_ids, sp_inverse_ids, sp_areas = np.unique(self.data, return_counts=True, return_inverse=True)

        keep = np.arange(sp_areas.size)
        if min_size:
            keep = keep[sp_areas > min_size]
            sp_areas = sp_areas[keep]
        if (max_num is not None) and (sp_areas.size > max_num):
            keep = keep[sp_areas.argsort()[::-1][:max_num]]

        lookup = np.full((sp_ids.size,), max_num, np.int64)
        lookup[keep] = np.arange(keep.size)
        new_data = lookup[sp_inverse_ids].reshape(self.data.shape)
        self.data = new_data

class TransformablePoint(TransformableData):
    @classmethod
    def load(cls, data, size=None):
        return cls(np.array(data[1:] if isinstance(data[0], str) else data, np.float64), size)

    def __init__(self, data, size=None):
        assert isinstance(data, np.ndarray), type(data)
        assert data.ndim == 2, data.shape
        super().__init__(data)

        self.size = size

    def scale(self, s):
        self.data[:, -2:] *= s
        if self.size is not None:
            x0, y0 = self.size
            self.size = [x0*s, y0*s]

    def resize(self, size):
        if self.size is None:
            raise RuntimeError(f"Need size")
        x, y = size
        x0, y0 = self.size
        self.data[:, -2] *= float(x) / x0
        self.data[:, -1] *= float(y) / y0
        self.size = size

    def mirror(self):
        if self.size is None:
            raise RuntimeError(f"Need size")
        self.data[:, -2] = self.size[0] - self.data[:, -2]

    def pad(self, top, bottom, left, right):
        self.data[:, -2] += left
        self.data[:, -1] += top

    def crop(self, top, bottom, left, right):
        self.data[:, -2] -= left
        self.data[:, -1] -= top

    def normalize(self):
        pass
    def denormalize(self):
        pass

    def embed(self, random_shift=0, point_size=1, ignore_class=255, size=None):
        if size is None:
            size = [int(xy + .5) for xy in self.size]
        return TransformableLabel2D(_embed_points(self.data, size, random_shift, point_size, ignore_class))

def _embed_points(
        point,
        size,
        random_shift=0,
        point_size=1,
        ignore_class=255):
    out = np.full(size[::-1], ignore_class, dtype=np.int64)
    for L, x, y in point:
        x, y = int(x + .5), int(y + .5)
        if random_shift:
            dx, dy = np.random.randint(-random_shift, random_shift+1, (2,))
            x += dx
            y += dy
        if (x < 0) or (x >= size[0]) or (y < 0) or (y >= size[1]):
            continue

        bgnx = max(x - point_size // 2, 0)
        bgny = max(y - point_size // 2, 0)
        endx = min(bgnx + point_size, size[0])
        endy = min(bgny + point_size, size[1])
        out[bgny:endy, bgnx:endx] = L
    return out
