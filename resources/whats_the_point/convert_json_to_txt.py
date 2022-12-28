import numpy as np
import os
import json


def format_label(labels):
    out = []
    for label in labels:
        y = label['y']
        x = label['x']
        if (x < 0) or (y < 0):
            continue
        out.append( ','.join([str(val) for val in [label['cls'], label['y'], label['x']]]) )
    return out

def process_data(filename, data_list, save_file):
    with open(filename) as f:
        data = json.load(f)

    with open(data_list) as f:
        names = [x.strip() for x in f.readlines()]

    all_lines = []

    for name in names:
        labels = data.get(name, None)
        line = [name]
        if labels is not None:
            line += format_label(labels)
        line = ' '.join(line)
        all_lines.append(line)

    with open(save_file, 'w') as f:
        f.write('\n'.join(all_lines))

if __name__ == '__main__':
    filename = './data/pascal2012_trainval_supp.json'

    process_data(
            filename,
            '../../core/data/VOC/resources/train_aug.txt',
            './train_aug_points_foreground.txt'
    )



