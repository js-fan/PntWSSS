import sys
sys.path.append('../../')
from core.utils import *
import multiprocessing as mp
import json


def collect_names(root):
    cities = os.listdir(root)
    results = []
    for city in cities:
        dirname = os.path.join(root, city)
        names = list(set(['_'.join(x.split('_')[:3]) for x in os.listdir(dirname)]))
        results += names
    assert len(results) == len(set(results))
    return results

root = '/home/junsong_fan/diskf/data/cityscape/gtFine/train'
fine_train = collect_names(root)

root = '/home/junsong_fan/diskf/data/cityscape/gtCoarse/train_extra'
coarse_train_extra = collect_names(root)

root = '/home/junsong_fan/diskf/data/cityscape/gtCoarse/train'
coarse_train = collect_names(root)

print('length:')
print([len(x) for x in [fine_train, coarse_train, coarse_train_extra]])

print('fine_train + coarse_train:')
print(len(set(fine_train + coarse_train)))

print('fine_train + coarse_train_extra:')
print(len(set(fine_train + coarse_train_extra)))

print('coarse_train + coarse_train_extra:')
print(len(set(coarse_train + coarse_train_extra)))

