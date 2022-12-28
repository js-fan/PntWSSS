import torch
import numpy as np
import cv2
import os
from ...utils import CS, normalize_image, meani_list

class CSPointDataset(torch.utils.data.Dataset):
    def __init__(self, image_root, data_list, segment_root, target_size,
            rand_crop=False, rand_mirror=False, rand_scale=None, return_src=False,
            downsample_label=1, superpixel_root=None, max_superpixel=1024,
            return_image_label=False):
        # DATA: {city_id0_id1: src}
        # collect images
        self.image_dict = {}
        for city in os.listdir(image_root):
            dirname= os.path.join(image_root,city)
            image_srcs = os.listdir(dirname)
            for src in image_srcs:
                self.image_dict['_'.join(src.split('_')[:3])] = os.path.join(dirname, src)

        # collect label
        if data_list is not None:
            with open(data_list) as f:
                data = [x.strip().split(' ') for x in f.readlines()]
 
            self.data_names = [x[0] for x in data]
            self.label_dict = {x[0]: [[int(L) for L in label.split(',')] for label in x[1:]] for x in data}

            missing_images = list(set(self.label_dict.keys()) - set(self.image_dict.keys()))
            missing_labels = list(set(self.image_dict.keys()) - set(self.label_dict.keys()))
            assert len(missing_images) == 0, "Miss the following {} images:\n{}".format(len(missing_images), missing_images)
            assert len(missing_labels) == 0, missing_labels
        else:
            self.data_names = list(self.image_dict.keys())
            self.label_dict = None

        # collect gt
        if segment_root is not None:
            self.gt_dict = {}
            for city in os.listdir(segment_root):
                dirname = os.path.join(segment_root, city)
                gt_srcs = [x for x in os.listdir(dirname) if x.endswith('labelIds.png')]
                for src in gt_srcs:
                    self.gt_dict['_'.join(src.split('_')[:3])] = os.path.join(dirname, src)
            missing_gts = list(set(self.image_dict.keys() - set(self.gt_dict.keys())))
            assert len(missing_gts) == 0, missing_gts
        else:
            self.gt_dict = None

        # collect superpixel
        if superpixel_root:
            self.sp_dict = {}
            sp_srcs = filter(lambda x: x.endswith('.png'), os.listdir(superpixel_root))
            for sp_src in sp_srcs:
                self.sp_dict[sp_src[:-4]] = os.path.join(superpixel_root, sp_src)
            missing_sps = list(set(self.image_dict.keys()) - set(self.sp_dict.keys()) )
            assert len(missing_sps) == 0, missing_sps
        else:
            self.sp_dict = None

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
        key = self.data_names[index]
        src = self.image_dict[key]
        lbl = self.label_dict[key] if self.label_dict is not None else None
        gt_src = self.gt_dict[key] if self.gt_dict is not None else None
        sp_src = self.sp_dict[key] if self.sp_dict is not None else None

        return_vals = fn_load_transform(src, lbl, gt_src, sp_src,
                self.target_size, self.rand_crop, self.rand_mirror, self.rand_scale,
                self.downsample_label, self.max_superpixel
        )
        #return_vals = list(filter(lambda x: x is not None, [img, lbl, gt, sp]))
        return_vals = [x for x in return_vals if x is not None]

        # image level label
        if self.return_image_label and (lbl is not None):
            image_label = torch.zeros((19,), dtype=torch.int64)
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
    img = cv2.imread(src)
    h, w = img.shape[:2]

    #has_lbl = True if len(lbl) > 0 else False
    has_lbl = False
    if lbl is not None:
        lbl_class = np.array([x[0] for x in lbl])
        lbl_coords = np.array([x[1:] for x in lbl]).astype(np.float32)
        # rescale
        lbl_coords *= float(h) / 1024.
        has_lbl = True

    has_gt = False
    if segment_src:
        gt = cv2.imread(segment_src, 0)
        assert gt.shape == (h, w), (gt.shape, img.shape, segment_src, src)
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
        gt = torch.from_numpy(CS.id2trainId[gt.ravel()].reshape(gt.shape).astype(np.int64))
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

