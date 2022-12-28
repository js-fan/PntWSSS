import torch
import numpy as np
import cv2
from pathlib import Path


_normalize_mean = (0.406, 0.456, 0.485)
_normalize_std = (0.225, 0.224, 0.229)

class PointDataset(torch.utils.data.Dataset):
    def __init__(self,
            image_root,
            image_suffix,
            point_label_file
            ):
        # Images
        if not isinstance(image_root, (list, tuple)):
            image_root = [image_root]
            image_suffix = [image_suffix]
            self._rtn_image_list = False
        else:
            self._rtn_image_list = True

        if isinstance(image_suffix, str):
            image_suffix = [image_suffix] * len(image_root)
        assert len(image_root) == len(image_suffix), \
                f"LENGTH not match: {len(image_root), len(image_suffix)}"

        images = [[x.resolve() for x in Path(_root).rglob('*' + _suffix)] \
                for _root, _suffix in zip(image_root, image_suffix)]
        self._num_images = len(images)
        if self._num_images > 1:
            assert all([len(_images) == len(images[0]) for _images in images[1:]]), \
                    f"LENGTH not match: {[len(_images) for _images in images]} for {image_root}, {image_suffix}"
        self.images = [sorted(_images, key=lambda x: x.name) for _images in images]

        # Points
        if point_label_file is not None:
            with open(point_label_file) as f:
                points = [x.strip().split(' ') for x in f.readlines()]
            assert len(points) == len(images[0])

            _mapInt = lambda x: [int(_val) for _val in x.split(',')]
            points = [pnt[0:1] + list(map(_mapInt, pnt[1:])) for pnt in points]
            self.points = sorted(points, key=lambda x: x[0])
        else:
            self.points = None

    def __len__(self):
        return len(self.images[0])

    def __getitem__(self, index):
        image, point = self.get_basic(index)
        size = image.shape[:2] if isinstance(image, np.ndarray) else image[0].shape[:, 2]
        point_map = self.embed_points(point, size)
        data = [image, point_map] if isinstance(image, np.ndarray) else (image + [point_map])
        return self.to_torch(data)

    def get_basic(self, index):
        if self._rtn_image_list:
            image = [cv2.imread(str(self.images[i][index])) for i in range(self._num_images)]
            self.last_image = [str(self.images[i][index]) for i in range(self._num_images)]
        else:
            image = cv2.imread(str(self.images[0][index]))
            self.last_image = str(self.images[0][index])
        point = [] if self.points is None else np.array(self.points[index][1:]).astype(np.int64)
        return image, point

    @classmethod
    def embed_points(cls,
            point,
            size,
            random_shift=0,
            point_size=1,
            ignore_class=255
            ):
        assert len(size) == 2, size
        return _embed_points(point, size, random_shift, point_size, ignore_class)

    @classmethod
    def trans_resize_short(cls,
            image,
            point,
            short
            ):
        if isinstance(image, np.ndarray):
            h, w = image.shape[:2]
        else:
            #NOTE: If multiple images, assume they are of the same size.
            h, w = image[0].shape[:2]

        scale = float(short) / min(h, w)
        if (scale == 1):
            return image, point
        return cls.trans_resize_scale(image, point, scale)

    @classmethod
    def trans_resize_scale(cls,
            image,
            point,
            scale
            ):
        images = [image] if isinstance(image, np.ndarray) else image

        h, w = images[0].shape[:2]
        h2, w2 = int(h * scale + .5), int(w * scale + .5)
        images = [cv2.resize(_image, (w2, h2)) for _image in images]

        if isinstance(image, np.ndarray):
            images = images[0]

        if len(point) > 0:
            point[:, 1:] = (point[:, 1:].astype(np.float32) * scale + .5).astype(np.int64)
        return images, point

    @classmethod
    def trans_crop(cls,
            image,
            point,
            size,
            random,
            pad=(104, 117, 124)
            ):
        if not isinstance(image, np.ndarray):
            assert len(pad) == len(image), f"NUM not match: numImage={len(image)}, numPad={len(pad)}"
            expect_pad_dims = [1 if (_image.ndim == 2) else 3 for _image in image]
            assert all([_exp_dim == len(_pad) for _exp_dim, _pad in zip(expect_pad_dims, pad)]), \
                    f"DIM not match: expected={expect_pad_dims}, given={[len(_pad) for _pad in pad]}, pad={pad}"
            images = image
        else:
            images = [image]
            pad = [pad]

        h, w = images[0].shape[:2]
        tgt_h, tgt_w = size
        pad_h, pad_w = max(tgt_h - h, 0), max(tgt_w - w, 0)
        if (pad_h > 0) or (pad_w > 0):
            images = [cv2.copyMakeBorder(_image,
                pad_h//2, pad_h-pad_h//2,
                pad_w//2, pad_w-pad_w//2,
                cv2.BORDER_CONSTANT,
                value=_pad
                ) for (_image, _pad) in zip(images, pad)]
            if len(point) > 0:
                point[:, 1:] += (pad_w//2, pad_h//2)

        cut_h, cut_w = h - tgt_h, w - tgt_w
        if random:
            bgn_h, bgn_w = np.random.randint(0, cut_h+1), np.random.randint(0, cut_w+1)
        else:
            bgn_h, bgn_w = cut_h // 2, cut_w // 2
        images = [_image[bgn_h:bgn_h+tgt_h, bgn_w:bgn_w+tgt_w] for _image in images]

        if isinstance(image, np.ndarray):
            images = images[0]

        if len(point) > 0:
            point[:, 1:] -= (bgn_w, bgn_h)
        return images, point

    @classmethod
    def trans_flip(cls,
            image,
            point
            ):
        images = [image] if isinstance(image, np.ndarray) else image
        images = [_image[:, ::-1] for _image in images]

        if len(point) > 0:
            point[:, 1] = images[0].shape[1] - 1 - point[:, 1]

        if isinstance(image, np.ndarray):
            images = images[0]
        return images, point

    @classmethod
    def normalize(cls,
            image
            ):
        assert (image.ndim == 3) and (image.shape[-1] == 3), image.shape
        return ((image.astype(float)/255. - _normalize_mean) / _normalize_std).astype(np.float32)

    @classmethod
    def denormalize(cls,
            image
            ):
        if isinstance(image, torch.Tensor):
            image = image.data.cpu().numpy()

        if (image.ndim == 2):
            return (image * 255.99).astype(np.uint8)

        assert image.ndim == 3, image.shape
        if image.shape[-1] != 3:
            assert image.shape[0] == 3, image.shape
            image = image.transpose(1, 2, 0)[..., ::-1]
        return (((image * _normalize_std) + _normalize_mean) * 255.).astype(np.uint8)

    @classmethod
    def to_torch(cls,
            image,
            normalize=True
            ):
        images = [image] if isinstance(image, np.ndarray) else image
        if normalize:
            images = [cls.normalize(_image) if (_image.ndim == 3) else _image for _image in images]

        images = [torch.from_numpy( (_image[..., ::-1].transpose(2, 0, 1).copy()) if (_image.ndim == 3) else _image ) \
                for _image in images]
        if isinstance(image, np.ndarray):
            images = images[0]

        return images

def _embed_points_py(
        point,
        size,
        random_shift=0,
        point_size=1,
        ignore_class=255):
    out = np.full(size, ignore_class, dtype=np.int64)
    for L, x, y in point:
        if random_shift:
            dx, dy = np.random.randint(-random_shift, random_shift+1, (2,))
            x += dx
            y += dy
        if (x < 0) or (x >= size[1]) or (y < 0) or (y >= size[0]):
            continue

        bgnx = max(x - point_size // 2, 0)
        bgny = max(y - point_size // 2, 0)
        endx = min(bgnx + point_size, size[1])
        endy = min(bgny + point_size, size[0])
        out[bgny:endy, bgnx:endx] = L
    return out

try:
    from dataset_utils import _embed_points
except:
    _embed_points = _embed_points_py


if __name__ == '__main__':
    dataset = PointDataset('../../data/cityscapes/leftImg8bit/train', '.png', '../../resources/labels/cityscapes/uniform_instance.txt')
    img, pnt = dataset.get_basic(0)
    img, pnt = dataset.trans_resize_short(img, pnt, 512)
    print(img.shape)
    lbl = dataset.embed_points(pnt, img.shape[:2])
    print(np.unique(lbl))


