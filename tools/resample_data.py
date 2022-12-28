import numpy as np
from pathlib import Path

def sample_labels(labels, N):
    Ls = [int(x.split(',')[0]) for x in labels]
    cnt = np.zeros((256,), np.int64)
    indices = []
    for i in np.random.permutation(len(Ls)):
        L = Ls[i]
        if cnt[L] >= N:
            continue
        cnt[L] += 1
        indices.append(i)
    return sorted([labels[i] for i in indices], key=lambda x: int(x.split(',')[0]))

def resample_points(src, dst, max_point_per_class):
    with open(src) as f:
        data = [x.strip().split(' ') for x in f.readlines()]

    data = [x[0:1] + sample_labels(x[1:], max_point_per_class) for x in data]
    data = [' '.join(x) for x in data]

    dst = Path(dst)
    dst.resolve().parents[0].mkdir(parents=True, exist_ok=True)
    dst.open('w').write('\n'.join(data))

if __name__ == '__main__':
    nPnt = 1
    resample_points(
            '../resources/labels/cityscapes/center_instance.txt',
            f'../resources/labels/cityscapes/center_instance_pnt{nPnt}.txt',
            nPnt)
