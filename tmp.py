import numpy as np
import os
import cv2

'''
def collect_names(root):
    cities = os.listdir(root)
    out = {}
    for city in cities:
        names = os.listdir(os.path.join(root, city))
        for name in names:
            prefix = '_'.join(name.split('_')[:3])
            out[prefix] = os.path.join(root, city, name)
    return out

root = '/home/junsong_fan/diskf/data/cityscape/leftImg8bit_jpg/train'
sp_root = '/home/junsong_fan/diskf/data/cityscape/leftImg8bit_scg_png/train'
data = collect_names(root)
np.random.seed(0)
palette = np.random.randint(0, 256, (1000, 3), np.uint8)

cnt = 0
out = []
for prefix, src in data.items():
    img = cv2.imread(src)
    sp = cv2.imread(os.path.join(sp_root, prefix+'.png'))
    assert img.shape[:2] == sp.shape[:2], (src, img.shape, sp.shape)
    sp = sp.astype(np.int64)
    sp = sp[..., 0] + sp[..., 1]*256 + sp[...,2]*65536
    sp_color = palette[sp.ravel()].reshape(sp.shape+(3,))
    out.append(np.hstack([img, sp_color]))
    cnt += 1
    if cnt >= 10:
        break

out = np.vstack(out)
cv2.imwrite('./tmp_cs_sp.jpg', out)
'''


def revise_index(sp, max_sp):
    h, w = sp.shape
    sp_ids, sp2, sp_areas = np.unique(sp, return_counts=True, return_inverse=True)
    if len(sp_ids) > max_sp:
        to_keep = sp_areas.argsort()[-max_sp:]
        lookup = np.full((len(sp_ids,),), max_sp)
        lookup[to_keep] = np.arange(max_sp)
        sp3 = lookup[sp2]

        print('sp')
        print(sp)

        print('ids, sp2, area, to_keep, lookup')
        print(sp_ids)
        print(sp_areas)
        print(to_keep)
        print(lookup)
        print(sp2.reshape(h, w))

        print('sp3')
        print(sp3.reshape(h, w))


revise_index(np.random.randint(0, 8, (7, 7)), 2)
