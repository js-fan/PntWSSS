import numpy as np
import cv2
import os


def sample_background(src, N=1):
    seg = cv2.imread(src, 0)
    h, w = seg.shape
    candidate = np.nonzero(seg.ravel() == 0)[0]
    if candidate.size == 0:
        return None

    indices = candidate[np.random.permutation(candidate.size)[:N]]
    out = []
    for idx in indices:
        ih = idx // w
        iw = idx % w
        out.append( ','.join([str(val) for val in [0, ih, iw]]) )
    return out

def process_data(src_file, dst_file, seed_root, N=1):
    with open(src_file) as f:
        data = [x.strip().split(' ') for x in f.readlines()]

    for i in range(len(data)):
        name = data[i][0]
        src = os.path.join(seed_root, name + '.png') 
        bg_labels = sample_background(src, N)
        if bg_labels is not None:
            data[i] += bg_labels

    data = [' '.join(line) for line in data]
    with open(dst_file, 'w') as f:
        f.write('\n'.join(data))

if __name__ == '__main__':
    src_file = './train_aug_points_foreground.txt'

    dst_file = './train_aug_points_gtBackground.txt'
    seed_root = '/home/junsong_fan/diskf/data/VOC2012/extra/SegmentationClassAug'
    process_data(src_file, dst_file, seed_root)
