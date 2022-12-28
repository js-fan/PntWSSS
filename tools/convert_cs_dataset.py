import sys
sys.path.append('../')
from pathlib import Path
import cv2
from core.utils import *
import multiprocessing as mp


def convert(src, dst):
    img = cv2.imread(src, 0)
    img = CS.id2trainId[img.ravel()].reshape(img.shape)
    cv2.imwrite(dst, img)

def main():
    root = '../data/cityscapes/gtFine/train'
    dst_root = '../data/cityscapes/gtFine_trainIds/train'

    files = [x.resolve() for x in Path(root).rglob("*_labelIds.png")]
    dst_root = Path(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    tasks = []
    for filename in files:
        tasks.append([str(filename), str(dst_root / filename.name.replace('_labelIds.png', '_trainIds.png'))])

    pool = mp.Pool(16)
    jobs = [pool.apply_async(convert, task) for task in tasks]
    [job.get() for job in jobs]

main()
