import torch
import numpy as np
import cv2
import os
from ...utils import VOC, normalize_image, meani_list

_curr_path = os.path.dirname(os.path.abspath(__file__))

class VOCPointDataset(torch.utils.data.Dataset):
    def __init__(self, image_root, label_file, segment_root, split, target_size,
            rand_crop=False, rand_mirror=False, rand_scale=None, return_src=False,
            downsample_label=1, superpixel_root=None, max_superpixel=1024,
            return_image_label=False,
            short_edge_scale=False):
        # data sub-split names
        assert split in ['train_aug', 'val', 'test']
        name_file = os.path.join(_curr_path, 'resources', split+'.txt')
        with open(name_file) as f:
            names = [x.strip() for x in f.readlines()]

        # image srcs
        self.data_images = [os.path.join(image_root, x+'.jpg') for x in names]
        for src in self.data_images:
            assert os.path.exists(src), src

        # point labels (opt.)
        if label_file is not None:
            assert os.path.exists(label_file), label_file
            with open(label_file) as f:
                labels_data = [x.strip().split(' ') for x in f.readlines()]
                labels_name = [x[0] for x in labels_data]
                labels_label = [[[float(v) for v in this.split(',')] for this in x[1:]] for x in labels_data]
                labels_dict = {name : label for name, label in zip(labels_name, labels_label)}
            self.data_labels = [labels_dict[name] for name in names]
        else:
            self.data_labels = None

        # segmentation maps (opt.)
        if segment_root is not None:
            self.data_segments = [os.path.join(segment_root, x+'.png') for x in names]
            for src in self.data_segments:
                assert os.path.exists(src), src
        else:
            self.data_segments = None

        # superpixels (opt.)
        if superpixel_root:
            self.data_superpixel = [os.path.join(superpixel_root, x+'.png') for x in names]
            for src in self.data_superpixel:
                assert os.path.exists(src), src
        else:
            self.data_superpixel = None

        # attrs
        self.target_size = target_size
        self.rand_crop = rand_crop
        self.rand_mirror = rand_mirror
        self.rand_scale = rand_scale
        self.return_src = return_src
        self.downsample_label = downsample_label
        self.max_superpixel = max_superpixel
        self.short_edge_scale = short_edge_scale
        self.return_image_label = return_image_label

    def __getitem__(self, index):
        src = self.data_images[index]
        lbl = self.data_labels[index] if self.data_labels is not None else None
        gt_src = self.data_segments[index] if self.data_segments is not None else None
        sp_src = self.data_superpixel[index] if self.data_superpixel is not None else None

        return_vals = fn_load_transform(src, lbl, gt_src, sp_src,
                self.target_size, self.rand_crop, self.rand_mirror, self.rand_scale,
                self.downsample_label, self.max_superpixel,
                self.short_edge_scale
        )
        return_vals = [x for x in return_vals if x is not None]

        # image level label
        if self.return_image_label and (lbl is not None):
            image_label = torch.zeros((21,), dtype=torch.int64)
            for lxy in lbl:
                image_label[int(lxy[0])] = 1
            return_vals.append(image_label)

        # image src
        if self.return_src:
            return_vals.append(src)
        return return_vals

    def __len__(self):
        return len(self.data_images)

def fn_load_transform(src, lbl, gt_src, superpixel_src,
        target_size, rand_crop, rand_mirror, rand_scale, downsample_label, max_superpixel,
        short_edge_scale):
    img = cv2.imread(src)
    h, w = img.shape[:2]

    has_lbl = False
    if lbl is not None:
        lbl_class = np.array([int(x[0]) for x in lbl])
        lbl_coords = np.array([x[1:] for x in lbl]).astype(np.float32)
        has_lbl = True

    has_gt = False
    if gt_src:
        gt = cv2.imread(gt_src, 0)
        assert gt.shape == (h, w), (gt.shape, img.shape, gt_src, src)
        has_gt = True

    has_sp = False
    if superpixel_src:
        sp = cv2.imread(superpixel_src)
        assert sp.shape[:2] == (h, w), (sp.shape, img.shape, superpixel_src, src)
        sp = sp.astype(np.int64)
        sp = sp[..., 0] + sp[..., 1] * 256 + sp[..., 2] * 65536
        has_sp = True

    if rand_scale is not None:
        scale = np.random.uniform(rand_scale[0], rand_scale[1])
        if short_edge_scale:
            if h < w:
                new_h = int(target_size[0] * scale + .5)
                new_w = int(float(w) / h * new_h + .5)
            else:
                new_w = int(target_size[1] * scale + .5)
                new_h = int(float(h) / w * new_w + .5)
            h, w = new_h, new_w
        else:
            h = int(h * scale + .5)
            w = int(w * scale + .5)
        img = cv2.resize(img, (w, h))

        if has_gt:
            gt = cv2.resize(gt, (w, h), interpolation=cv2.INTER_NEAREST)
        if has_lbl:
            lbl_coords *= scale
        if has_sp:
            sp = cv2.resize(sp, (w, h), interpolation=cv2.INTER_NEAREST)

    if rand_mirror and np.random.rand() > 0.5:
        img = img[:, ::-1]
        if has_gt:
            gt = gt[:, ::-1]
        if has_lbl:
            lbl_coords[:, 1] = w - 1 - lbl_coords[:, 1]
        if has_sp:
            sp = sp[:, ::-1]

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
            if has_lbl:
                lbl_coords += np.array([pt, pl]).reshape(1, -1)
            if has_sp:
                sp = cv2.copyMakeBorder(sp, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=int(sp.max())+1)

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
        if has_lbl:
            lbl_coords -= np.array([bh, bw]).reshape(1, -1)
        if has_sp:
            sp = sp[bh : bh+target_size[0], bw : bw+target_size[1]]

    img = torch.from_numpy(normalize_image(img)[..., ::-1].transpose(2, 0, 1).copy())
    ih, iw = img.size()[1:]

    if has_gt:
        if downsample_label != 1:
            dh, dw = (ih // downsample_label) + (ih % 2), (iw // downsample_label) + (iw % 2)
            gt = cv2.resize(gt, (dw, dh), interpolation=cv2.INTER_NEAREST)
        gt = torch.from_numpy(gt.astype(np.int64))
    else:
        gt = None

    if has_lbl:
        if downsample_label != 1:
            h, w = (ih // downsample_label) + (ih % 2), (iw // downsample_label) + (iw % 2)
            lbl_coords /= downsample_label
        else: 
            h, w = ih, iw
        lbl = torch.full((h, w), 255, dtype=torch.int64)
        lbl_coords = lbl_coords.astype(np.int64)
        for c, (y, x) in zip(lbl_class, lbl_coords):
            if (x >= 0) and (y >= 0) and (x < w) and (y < h):
                lbl[y, x] = c
    else:
        lbl = None

    if has_sp:
        if downsample_label != 1:
            dh, dw = (ih // downsample_label) + (ih % 2), (iw // downsample_label) + (iw % 2)
            sp = cv2.resize(sp, (dw, dh), interpolation=cv2.INTER_NEAREST)

        # check if |sp| < max_superpixel
        sp_ids, sp2, sp_areas = np.unique(sp, return_counts=True, return_inverse=True)
        if len(sp_ids) > max_superpixel:
            to_keep = sp_areas.argsort()[-max_superpixel:]
            lookup = np.full((sp_ids.size,), max_superpixel, np.int64)
            lookup[to_keep] = np.arange(max_superpixel)
            sp2 = lookup[sp2]
        sp = sp2.reshape(sp.shape)
        sp = torch.from_numpy(sp)
    else:
        sp = None

    return img, lbl, gt, sp
