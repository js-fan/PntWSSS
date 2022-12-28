import torch
import cv2
import os
import numpy as np
from ...utils import CS, normalize_image, meani_list

class CSClassDataset(torch.utils.data.Dataset):
    def __init__(self, image_root, label_list, target_size,
            rand_crop=False, rand_mirror=False, rand_scale=None, return_src=False,
            segment_root=None, downsample_label=1, use_city_name=False):
        with open(label_list) as f:
            data = [x.strip().split(' ') for x in f.readlines()]
        self.data_names = [x[0] for x in data]

        # label list
        self.label_list = [[int(L) for L in x[1:]] for x in data]

        # image list
        self.image_list = []
        for name in self.data_names:
            if use_city_name:
                city = name.split('_')[0]
                img_src = os.path.join(image_root, city, name + '.jpg')
            else:
                img_src = os.path.join(image_root, name + '.jpg')
            assert os.path.exists(img_src), img_src

            self.image_list.append(img_src)

        # gt list
        if segment_root is not None:
            self.segment_list = []
            for name in self.data_names:
                gt_src = os.path.join(segment_root, name + '.png')
                assert os.path.exists(gt_src), gt_src

                self.segment_list.append(gt_src)
        else:
            self.segment_list = None

        # attrs
        self.target_size = target_size
        self.rand_crop = rand_crop
        self.rand_mirror = rand_mirror
        self.rand_scale = rand_scale
        self.downsample_label = downsample_label
        self.return_src = return_src

    def __getitem__(self, index):
        src = self.image_list[index]
        lbl_ids = self.label_list[index]
        lbl = torch.zeros((19,), dtype=torch.float32)
        lbl[lbl_ids] = 1
        gt_src = self.segment_list[index] if self.segment_list is not None else None

        img, gt = fn_load_transform(src, gt_src,
                self.target_size, self.rand_crop, self.rand_mirror, self.rand_scale,
                self.downsample_label
        )
        return_vals = list(filter(lambda x: x is not None, [img, lbl, gt]))
        if self.return_src:
            return_vals.append(src)
        return return_vals

    def __len__(self):
        return len(self.data_names)

def fn_load_transform(src, gt_src,
        target_size, rand_crop, rand_mirror, rand_scale, downsample_label):
    img = cv2.imread(src)
    h, w = img.shape[:2]

    has_gt = False
    if gt_src:
        gt = cv2.imread(gt_src, 0)
        assert gt.shape == (h, w), (gt.shape, img.shape, gt_src, src)
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
        pt = ph // 2
        pb = ph - pt
        pl = pw // 2
        pr = pw - pl
        if ph > 0 or pw > 0:
            img = cv2.copyMakeBorder(img, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=meani_list)
            if has_gt:
                gt = cv2.copyMakeBorder(gt, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=255)
        h, w = img.shape[:2]

        if rand_crop:
            bh = np.random.randint(0, h - target_size[0] + 1)
            bw = np.random.randint(0, w - target_size[1] + 1)
        else:
            bh = (h - target_size[0]) // 2
            bw = (w - target_size[1]) // 2

        img = img[bh : bh+target_size[0], bw : bw + target_size[1]]
        if has_gt:
            gt = gt[bh : bh+target_size[0], bw : bw+target_size[1]]

    img = torch.from_numpy(normalize_image(img)[..., ::-1].transpose(2, 0, 1).copy())
    ih, iw = img.size()[1:]

    if has_gt:
        if downsample_label != 1:
            dh, dw = (ih // downsample_label) + (ih % 2), (iw // downsample_label) + (iw % 2)
            gt = cv2.resize(gt, (dw, dh), interpolation=cv2.INTER_NEAREST)
        gt = torch.from_numpy(CS.id2trainId[gt.ravel()].reshape(gt.shape).astype(np.int64))
    else:
        gt = None

    return img, gt
