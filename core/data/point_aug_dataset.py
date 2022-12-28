import torch
import numpy as np
import cv2
from .point_dataset import PointDataset

def _as_list(x):
    return x if isinstance(x, list) else [x]

class PointAugDataset(PointDataset):
    def __init__(self,
            image_root,
            point_label_file, 
            image_size,
            rand_short=False,
            rand_crop=False,
            rand_mirror=False,
            rand_point_shift=0,
            point_size=1,
            pad_value=(104, 117, 124),
            image_suffix=".png"
            ):
        super().__init__(image_root, image_suffix, point_label_file)

        self.image_size = image_size
        self.rand_short = rand_short
        self.rand_crop = rand_crop
        self.rand_mirror = rand_mirror
        self.rand_point_shift = rand_point_shift
        self.point_size = point_size

        if isinstance(image_root, (list, tuple)):
            if not isinstance(pad_value, list):
                pad_value = [pad_value] * len(image_root)
        self.pad_value = pad_value

    def get_basic(self, index):
        data = super().get_basic(index)
        if self.rand_short:
            data = self.trans_resize_short(*data,
                    short=np.random.randint(self.rand_short[0], max(self.rand_short[1], self.rand_short[0]+1)))

        data = self.trans_crop(*data,
                size=self.image_size,
                random=self.rand_crop,
                pad=self.pad_value)

        if self.rand_mirror and np.random.rand() > 0.5:
            data = self.trans_flip(*data)
        return data

    def to_torch(self, data):
        images, point = data
        if self.points is not None:
            images = _as_list(images)
            point_map = self.embed_points(point,
                    images[0].shape[:2],
                    self.rand_point_shift,
                    self.point_size,
                    ignore_class=255)
            images.append(point_map)
        data = super().to_torch(images)
        return data

    def __getitem__(self, index):
        data = self.get_basic(index)
        data = self.to_torch(data)
        return data

class PointAugDatasetWithSP(PointAugDataset):
    def __init__(self,
            image_root,
            superpixel_root,
            point_label_file, 
            image_size,
            rand_short=False,
            rand_crop=False,
            rand_mirror=False,
            rand_point_shift=0,
            point_size=1,
            pad_value=(104, 117, 124),
            image_suffix=".png",
            max_superpixel=1024,
            post_resize_superpixel=False,
            rtn_src=False
            ):
        self.has_sp = bool(superpixel_root)
        if self.has_sp:
            image_root = _as_list(image_root) + [superpixel_root]
            if not isinstance(pad_value, list):
                pad_value = [pad_value] * (len(image_root) - 1)
            assert len(pad_value) == (len(image_root) - 1), \
                    f"LENGTH not math: {len(pad_value)}, {len(image_root)-1}"
            pad_value = pad_value + [(255, 255, 255)]

        super().__init__(
                image_root,
                point_label_file, 
                image_size,
                rand_short,
                rand_crop,
                rand_mirror,
                rand_point_shift,
                point_size,
                pad_value,
                image_suffix
                )
        self.max_superpixel = max_superpixel
        self.post_resize_superpixel = [int(post_resize_superpixel*x) for x in image_size] \
            if isinstance(post_resize_superpixel, (float, int)) else post_resize_superpixel
        self.rtn_src = rtn_src

    def process_superpixel(self, sp):
        if self.post_resize_superpixel:
            sp = cv2.resize(sp, self.post_resize_superpixel[::-1])
        sp = sp.astype(np.int64)
        sp = sp[..., 0] + sp[..., 1] * 256 + sp[..., 2] * 65536
        sp_ids, sp_inverse_ids, sp_areas = np.unique(sp, return_counts=True, return_inverse=True)

        keep = np.arange(sp_areas.size)[sp_areas > 10]
        sp_areas = sp_areas[keep]
        if sp_areas.size > self.max_superpixel:
            keep = keep[sp_areas.argsort()[::-1][:self.max_superpixel]]

        lookup = np.full((sp_ids.size,), self.max_superpixel, np.int64)
        lookup[keep] = np.arange(keep.size)
        sp2 = lookup[sp_inverse_ids].reshape(sp.shape)
        return sp2

    def __getitem__(self, index):
        data = self.get_basic(index)
        if self.has_sp:
            images, point = data
            images[-1] = self.process_superpixel(images[-1])
            data = [images, point]

        data = self.to_torch(data)
        if self.rtn_src:
            data = _as_list(data) + [self.last_image]
        return data

if __name__ == '__main__':
    dataset = PointAugDatasetWithSP(
            [
                '../../data/cityscapes/leftImg8bit/train',
                '../../data/cityscapes/gtFine/train'
            ],
            '../../data/cityscapes/leftImg8bit_mcg_png/train',
            '../../resources/labels/cityscapes/uniform_instance.txt',
            image_suffix=['.png', '_color.png', '.png'],
            image_size=[512, 1024],
            rand_short=[512, 768],
            rand_crop=True,
            rand_mirror=True,
            post_resize_superpixel=[512, 1024]
            )
    data = dataset[0]
    print([x.shape for x in data])
