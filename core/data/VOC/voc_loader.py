import torch
import numpy as np
import cv2
import os
from ...utils import VOC, normalize_image, meani_list

_curr_path = os.path.dirname(os.path.abspath(__file__))

class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, image_root, label_root, segment_root, split, target_size,
            rand_crop=False, rand_mirror=False, rand_scale=None, return_src=False):
        assert split in ['train_aug', 'val', 'test']
        name_file = os.path.join(_curr_path, 'resources', split+'.txt')

        with open(name_file) as f:
            names = [x.strip() for x in f.readlines()]

        self.data_images = [os.path.join(image_root, x+'.jpg') for x in names]
        self.data_labels = [os.path.join(label_root, x+'.xml') for x in names] if label_root else None
        self.data_segment = [os.path.join(segment_root, x+'.png') for x in names] if segment_root else None

        self.target_size = target_size
        self.rand_crop = rand_crop
        self.rand_mirror = rand_mirror
        self.rand_scale = rand_scale
        self.return_src = return_src

    def __getitem__(self, index):
        src = self.data_images[index]
        segment_src = self.data_segment[index] if self.data_segment else None
        img, gt = fn_load_transform(src, segment_src, self.target_size, self.rand_crop, self.rand_mirror, self.rand_scale)

        lbl = None
        if self.data_labels is not None:
            lbl = torch.zeros((20,), dtype=torch.int64)
            lbl[VOC.get_annotation(self.data_labels[index])] = 1

        return_vals = list(filter(lambda x: x is not None, [img, lbl, gt]))
        if self.return_src:
            return_vals.append(src)

        return return_vals

    def __len__(self):
        return len(self.data_images)


def fn_load_transform(src, segment_src, target_size, rand_crop, rand_mirror, rand_scale):
    # rgb
    img = cv2.imread(src)[..., ::-1]
    h, w = img.shape[:2]

    has_gt = False
    if segment_src:
        gt = cv2.imread(segment_src, 0)
        assert gt.shape == (h, w), (gt.shape, img.shape, segment_src, src)
        has_gt = True

    if rand_scale is not None:
        scale = np.random.uniform(rand_scale[0], rand_scale[1])
        h = int(h * scale + .5)
        w = int(w * scale + .5)
        img = cv2.resize(img, (w, h))

        if has_gt:
            gt = cv2.resize(gt, (w, h), interpolation=cv2.INTER_NEAREST)

    if rand_mirror and np.random.rand() > 0.5:
        img = img[:, ::-1]
        if has_gt:
            gt = gt[:, ::-1]

    if target_size is not None:
        ph = max(target_size[0] - h, 0)
        pw = max(target_size[1] - w, 0)
        if ph > 0 or pw > 0:
            img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=meani_list)
            if has_gt:
                gt = cv2.copyMakeBorder(gt, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=255)
        h, w = img.shape[:2]

        if rand_crop:
            bh = np.random.randint(0, h - target_size[0] + 1)
            bw = np.random.randint(0, w - target_size[1] + 1)
        else:
            bh = (h - target_size[0]) // 2
            bw = (w - target_size[0]) // 2

        img = img[bh : bh+target_size[0], bw : bw + target_size[1]]
        if has_gt:
            gt = gt[bh : bh+target_size[0], bw : bw+target_size[1]]

    img = torch.from_numpy(normalize_image(img).transpose(2, 0, 1))
    if has_gt:
        gt = torch.from_numpy(gt.astype(np.int64))
        return img, gt
    return img, None

