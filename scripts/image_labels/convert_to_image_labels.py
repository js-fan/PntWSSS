import cv2
import numpy as np
import os
import sys
import multiprocessing as mp
sys.path.append('../../')
from core.utils import *

def get_image_labels(gt_src):
    gt = cv2.imread(gt_src, 0)
    unique_ids = np.unique(gt)
    labels = [str(x) for x in CS.id2trainId[unique_ids] if x != 255]
    name = os.path.basename(gt_src).rsplit('.', 1)[0]
    return ' '.join([name] + labels)

def process_all_patches(gt_root, dst_file):
    srcs = os.listdir(gt_root)
    srcs.sort()

    # print(get_image_labels(os.path.join(gt_root, srcs[0]))

    pool = mp.Pool(56)
    jobs = [pool.apply_async(get_image_labels, (os.path.join(gt_root, src),)) for src in srcs]
    res = [job.get() for job in jobs]

    if not os.path.exists(os.path.dirname(dst_file)):
        os.makedirs(os.path.dirname(dst_file))
    with open(dst_file, 'w') as f:
        f.write('\n'.join(res))

def get_image_labels_raw(gt_src):
    gt = cv2.imread(gt_src, 0)
    unique_ids = np.unique(gt)
    labels = [str(x) for x in CS.id2trainId[unique_ids] if x != 255]
    name = '_'.join(os.path.basename(gt_src).split('_')[:3]) + '_leftImg8bit'
    return ' '.join([name] + labels)

def process_all_raw(gt_root, dst_file):
    cities = os.listdir(gt_root)
    srcs = []
    for city in cities:
        srcs += [os.path.join(gt_root, city, x) for x in os.listdir(os.path.join(gt_root, city)) if x.endswith('labelIds.png')]
    srcs.sort()

    pool = mp.Pool(56)
    jobs = [pool.apply_async(get_image_labels_raw, (os.path.join(gt_root, src),)) for src in srcs]
    res = [job.get() for job in jobs]

    if not os.path.exists(os.path.dirname(dst_file)):
        os.makedirs(os.path.dirname(dst_file))
    with open(dst_file, 'w') as f:
        f.write("\n".join(res))

if __name__ == '__main__':
    #process_all_patches('/home/junsong_fan/diskf/data/cityscape/patches/gtFine/train',
    #        './gtFine_patch_400x500_n15_imageLabels.txt')

    process_all_raw('/home/junsong_fan/diskf/data/cityscape/gtFine/train',
            './gtFine_1024x2048_imageLabels.txt')

