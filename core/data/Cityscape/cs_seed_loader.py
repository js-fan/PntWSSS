import torch
import numpy as np
import cv2
import os
from ...utils import CS, normalize_image, meani_list

class CSSeedDataset(torch.utils.data.Dataset):
    def __init__(self, image_root, seed_root_list, target_size,
            rand_crop=False, rand_mirror=False, rand_scale=None, return_src=False, downsample_label=1,
            superpixel_root=None, max_superpixel=1024):
        # DATA: {city_id0_id1: src}
        # collect images
        self.image_dict = {}
        self.data_names = []
        for city in os.listdir(image_root):
            dirname= os.path.join(image_root,city)
            image_srcs = os.listdir(dirname)
            for src in image_srcs:
                self.image_dict['_'.join(src.split('_')[:3])] = os.path.join(dirname, src)
                self.data_names.append('_'.join(src.split('_')[:3]))

        # collect seed
        if seed_root_list is not None:
            if not isinstance(seed_root_list, (list, tuple)):
                seed_root_list = [seed_root_list]
            self.seed_dict_list = []
            for seed_root in seed_root_list:
                self.seed_dict_list.append({})
                for city in os.listdir(seed_root):
                    dirname = os.path.join(seed_root, city)
                    seed_srcs = [x for x in os.listdir(dirname) if x.endswith('labelIds.png')]
                    for src in seed_srcs:
                        self.seed_dict_list[-1]['_'.join(src.split('_')[:3])] = os.path.join(dirname, src)
                missing_seeds = list( set(self.image_dict.keys()) - set(self.seed_dict_list[-1].keys()) )
                assert len(missing_seeds) == 0, missing_seeds
        else:
            self.seed_dict_list = None

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
    
    def __getitem__(self, index):
        key = self.data_names[index]
        src = self.image_dict[key]
        if self.seed_dict_list is None:
            seed_srcs = None
        else:
            seed_srcs = [seed_dict[key] for seed_dict in self.seed_dict_list]
        sp_src = self.sp_dict[key] if self.sp_dict is not None else None

        return_vals = fn_load_transform(src, seed_srcs, sp_src,
                self.target_size, self.rand_crop, self.rand_mirror, self.rand_scale, self.downsample_label,
                self.max_superpixel)
        return_vals = list(filter(lambda x: x is not None, return_vals))
        if self.return_src:
            return_vals.append(src)
        return return_vals
    
    def __len__(self):
        return len(self.image_dict)

def fn_load_transform(src, seed_srcs, superpixel_src,
        target_size, rand_crop, rand_mirror, rand_scale, downsample_label, max_superpixel):
    # img
    img = cv2.imread(src)
    h, w = img.shape[:2]

    # seed
    has_seed = False
    if seed_srcs:
        seeds = []
        for seed_src in seed_srcs:
            seed = cv2.imread(seed_src, 0)
            assert seed.shape == (h, w), (seed.shape, img.shape, seed_src, src)
            seeds.append(seed)
        has_seed = True
        num_seed = len(seeds)

    # sp
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

        if has_seed:
            for i in range(num_seed):
                seeds[i] = cv2.resize(seeds[i], (w, h), interpolation=cv2.INTER_NEAREST)
        if has_sp:
            sp = cv2.resize(sp, (w, h), interpolation=cv2.INTER_NEAREST)

    if rand_mirror and np.random.rand() > 0.5:
        img = img[:, ::-1]
        if has_seed:
            for i in range(num_seed):
                seeds[i] = seeds[i][:, ::-1]
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
            if has_seed:
                for i in range(num_seed):
                    seeds[i] = cv2.copyMakeBorder(seeds[i], pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=255)
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
        if has_seed:
            for i in range(num_seed):
                seeds[i] = seeds[i][bh : bh+target_size[0], bw : bw+target_size[1]]
        if has_sp:
            sp = sp[bh : bh+target_size[0], bw : bw+target_size[1]]

    img = torch.from_numpy(normalize_image(img)[..., ::-1].transpose(2, 0, 1).copy())
    ih, iw = img.size()[1:]

    if has_seed:
        if downsample_label != 1:
            dh, dw = (ih // downsample_label) + (ih % 2), (iw // downsample_label) + (iw % 2)
            for i in range(num_seed):
                seeds[i] = cv2.resize(seeds[i], (dw, dh), interpolation=cv2.INTER_NEAREST)
        for i in range(num_seed):
            seeds[i] = torch.from_numpy(CS.id2trainId[seeds[i].ravel()].reshape(seeds[i].shape).astype(np.int64))
    else:
        seeds = []

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
        sp = []

    return [img] + seeds + sp

