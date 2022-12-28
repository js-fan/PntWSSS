import numpy as np
import os

data = [
        '/home/junsong_fan/PointSeg/scripts/point_labels/Cityscape_1point_0ignore_uniform.txt',
        '/home/junsong_fan/PointSeg/scripts/point_labels/Cityscape_ins_point_uniform.txt',
        '/home/junsong_fan/PointSeg/resources/whats_the_point/train_aug_points_gtBackground.txt',
        '/home/junsong_fan/diskf/data/ADE20k/scripts/ADE20k_points.txt',
]

num_points = []

for filename in data:
    with open(filename) as f:
        d = [x.strip().split(' ') for x in f.readlines()]
        nPoints = np.array([len(x) - 1 for x in d])
        print(filename)
        print(nPoints.size, nPoints.sum(), nPoints.mean(), nPoints.min(), nPoints.max())




