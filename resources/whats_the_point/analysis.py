import numpy as np
import os
import json

filename = './data/pascal2012_trainval_main.json'
#filename = './data/pascal2012_trainval_supp.json'

def parse_label(labels):
    out = []
    for label in labels:
        out.append([
            label['cls'],
            label['y'],
            label['x'],
            label['rank']
        ]) 
    if len(out) == 0:
        return None
    return np.array(out)

with open(filename) as f:
    data = json.load(f)

keys = list(data.keys())
print(len(data))

k = keys[0]
#num_points = [len(x) for x in data.items()]
#print(len(num_points), sum(num_points) / len(num_points))

labels = []
for i, (k, label) in enumerate(data.items()):
    label = parse_label(label)
    if label is not None:
        labels.append(label)
    #if i >= 10: break

labels = np.vstack(labels)
print(labels.shape)
print(labels.min(axis=0), labels.max(axis=0))
print(np.unique(labels[:, 3]))
