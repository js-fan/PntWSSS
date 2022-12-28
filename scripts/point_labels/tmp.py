import cv2
import numpy as np
import os
import sys
import numba as nb
sys.path.append('../..')
from core.utils import *

def draw_point_label_demo():
    image_root = '/home/junsong_fan/diskf/data/cityscape/leftImg8bit_jpg/train'
    point_file = './Cityscape_1point_0ignore_uniform.txt'
    dst_root = './point_label_examples'
    
    targets = ['aachen_000002_000019', 'aachen_000000_000019', 'aachen_000001_000019']
    
    with open(point_file) as f:
        data = [x.strip().split(' ') for x in f.readlines()]
    
    if not os.path.exists(dst_root):
        os.makedirs(dst_root)
    
    data = {x[0]: x[1:] for x in data}
    
    for key in targets:
        img = cv2.imread(os.path.join(image_root, key.split('_')[0], key + '_leftImg8bit.jpg'))
        assert img is not None, key
    
        label_names = []
        for lhw in data[key]:
            lhw = [int(x) for x in lhw.split(',')]
            L = lhw[0]
            scale = 1024./1024.
            coords = (np.array(lhw[1:]) * scale).astype(np.int64)
    
            color = [int(x) for x in CS.palette[L]]
            #cv2.drawMarker(img, (coords[1], coords[0]), (255, 255, 255), cv2.MARKER_CROSS, 55, 9)
            #cv2.drawMarker(img, (coords[1], coords[0]), color, cv2.MARKER_CROSS, 50, 5)
            cv2.drawMarker(img, (coords[1], coords[0]), (255, 255, 255), cv2.MARKER_CROSS, 70, 11)
            cv2.drawMarker(img, (coords[1], coords[0]), color, cv2.MARKER_CROSS, 65, 7)
    
            label_names.append(CS.trainId2name[L])
    
        print(key, ', '.join(label_names))
        cv2.imwrite(os.path.join(dst_root, key + '.jpg'), img)

@nb.jit()
def get_sp_edge(sp, size=3):
    h, w = sp.shape
    out = np.zeros_like(sp)
    half_size = size // 2
    for i in range(h):
        for j in range(w):
            if out[i, j] == 1:
                continue

            val_i = sp[i, j]
            for m in range(-half_size, half_size + 1):
                for n in range(-half_size, half_size + 1):
                    pi = i + m
                    pj = j + n
                    if (pi < 0) or (pj < 0) or (pi >= h) or (pj >= w):
                        continue
                    val_p = sp[pi, pj]
                    if (val_i != val_p):
                        out[i, j] = 1
                        out[pi, pj] = 1
                        break
                if out[i, j] == 1:
                    break
    return out

def draw_superpixel_demo():
    image_root = '/home/junsong_fan/diskf/data/cityscape/leftImg8bit_jpg/train'
    sp_root = '/home/junsong_fan/diskf/data/cityscape/leftImg8bit_scg_png/train'
    dst_root = './superpixel_examples'

    targets = ['aachen_000002_000019', 'aachen_000000_000019', 'aachen_000001_000019']

    for key in targets:
        img = cv2.imread(os.path.join(image_root, key.split('_')[0], key + '_leftImg8bit.jpg'))
        sp = cv2.imread(os.path.join(sp_root, key + '.png')).astype(np.int64)
        sp = sp[..., 0] + sp[..., 1] * 256 + sp[..., 2] * 65536
        print(key, sp.max())
        edge = get_sp_edge(sp, 7)
        img[edge > 0] = np.uint8([0, 0, 255])

        imwrite(os.path.join(dst_root, key + '.jpg'), img)

#draw_superpixel_demo()
draw_point_label_demo()

