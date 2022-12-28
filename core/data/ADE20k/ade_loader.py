import torch
import numpy as np
import cv2
import os
from ...utils import normalize_image, meani_list

class ADEPointDataset(torch.utils.data.Dataset):
    def __init__(self, image_root, data_list, segment_root, target_size,
            rand_crop=False, rand_mirror=False, rand_scale=None, return_src=False,
            downsample_label=1, superpixel_root=None, max_superpixel=1024,
            return_image_label=False):

        # collect label
        if data_list is not None:
            with open(data_list) as f:
                data = [x.strip().split(' ') for x in f.readlines()]
            self.data_names = [x[0] for x in data]
            self.label_dict = {x[0]: [[int(L) for L in label.split(',')] for label in x[1:]] for x in data}

            empty_images = [k for k, v in self.label_dict.items() if len(v) == 0]
            for k in empty_images:
                del self.data_names[self.data_names.index(k)]
                del self.label_dict[k]
        else:
            self.data_names = [x.rsplit('.', 1)[0] for x in os.listdir(image_root) if x.endswith('.jpg')]
            self.label_dict = None

        # collect images
        self.image_root = image_root

        # collect gt
        if segment_root:
            self.gt_root = segment_root
        else:
            self.gt_root = None
        #assert self.gt_root is None, 'ADE20k gt loader not implemented yet.'

        self.sp_root = superpixel_root

        # attrs
        self.downsample_label = downsample_label
        self.target_size = target_size
        self.rand_crop = rand_crop
        self.rand_mirror = rand_mirror
        self.rand_scale = rand_scale
        self.return_src = return_src
        self.max_superpixel = max_superpixel
        self.return_image_label = return_image_label

    def __getitem__(self, index):
        name = self.data_names[index]
        src = os.path.join(self.image_root, name + '.jpg')
        lbl = self.label_dict[name] if self.label_dict is not None else None
        gt_src = os.path.join(self.gt_root, name + '.png') if self.gt_root is not None else None
        sp_src = os.path.join(self.sp_root, name + '.png') if self.sp_root is not None else None

        return_vals = fn_load_transform(src, lbl, gt_src, sp_src,
                self.target_size, self.rand_crop, self.rand_mirror, self.rand_scale,
                self.downsample_label, self.max_superpixel
        )
        return_vals = [x for x in return_vals if x is not None]

        # image level label
        if self.return_image_label and (lbl is not None):
            image_label = torch.zeros((150,), dtype=torch.int64)
            for lxy in lbl:
                image_label[int(lxy[0])] = 1
            return_vals.append(image_label)

        # image src
        if self.return_src:
            return_vals.append(src)
        return return_vals

    def __len__(self):
        return len(self.data_names)

def fn_load_transform(src, lbl, segment_src, superpixel_src,
        target_size, rand_crop, rand_mirror, rand_scale, downsample_label, max_superpixel):
    # basic load
    img = cv2.imread(src)
    h, w = img.shape[:2]

    has_lbl = False
    if lbl is not None:
        lbl_class = np.array([x[0] for x in lbl])
        lbl_coords = np.array([x[1:] for x in lbl]).astype(np.float32)
        has_lbl = True

    has_gt = False
    if segment_src:
        gt = cv2.imread(segment_src, 0)
        assert gt.shape == (h, w), (gt.shape, img.shape, segment_src, src)
        has_gt= True

    has_sp = False
    if superpixel_src:
        sp = cv2.imread(superpixel_src)
        assert sp.shape[:2] == (h, w), (sp.shape, img.shape, superpixel_src, src)
        sp = sp.astype(np.int64)
        sp = sp[..., 0] + sp[..., 1] * 256 + sp[..., 2] * 65536
        has_sp = True

    # scale
    if rand_scale is not None:
        scale = np.random.uniform(rand_scale[0], rand_scale[1])
        h = int(h * scale + .5)
        w = int(w * scale + .5)
        img = cv2.resize(img, (w, h))

        if has_gt:
            gt = cv2.resize(gt, (w, h), interpolation=cv2.INTER_NEAREST)
        if has_lbl:
            lbl_coords *= scale
        if has_sp:
            sp = cv2.resize(sp, (w, h), interpolation=cv2.INTER_NEAREST)

    # mirror
    if rand_mirror and np.random.rand() > 0.5:
        img = img[:, ::-1]
        if has_gt:
            gt = gt[:, ::-1]
        if has_lbl:
            lbl_coords[:, 1] = w - 1 - lbl_coords[:, 1]
        if has_sp:
            sp = sp[:, ::-1]

    # pad before cropping
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

        # crop
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

    # convert to torch.Tensor
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
