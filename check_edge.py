import numpy as np
import cv2
from pathlib import Path
from core.utils import *
import os
import matplotlib.pyplot as plt

#root = './snapshots/point_base/DeeplabV2_VGG16/cache/trainset'
root = './snapshots/point_pmask2/DeeplabV2_VGG16/cache/trainset'

names = [x.name.split('.png', 1)[0] for x in Path(root).rglob("*.png")]
img_root = './data/VOC2012/JPEGImages'
gt_root = './data/VOC2012/extra/SegmentationClassAug'
sp_root = './data/VOC2012/superpixel/mcg_png'
print(len(names))
np.random.shuffle(names)


def draw_sp(sp, img=None, c=(0, 0, 255)):
    kernel = np.ones((3, 3), np.float32)
    kernel[1, 1] = -8
    edge = cv2.filter2D(sp.astype(np.float32), -1, kernel)
    edge = edge != 0
    assert edge.shape == sp.shape, (edge.shape, sp.shape)

    if img is None:
        out = np.zeros(sp.shape+(3,), np.uint8).reshape(sp.size, 3)
    else:
        out = cv2.resize(img, sp.shape[::-1])
    out = out.reshape(sp.size, 3)
    out[edge.ravel()] = c
    out = out.reshape(sp.shape+(3,))
    return out

def sample(name):
    img = cv2.imread(os.path.join(img_root, name + '.jpg'))
    gt = cv2.imread(os.path.join(gt_root, name + '.png'), 0).astype(np.int32)
    gt_color = VOC.palette[gt.ravel()].reshape(gt.shape+(3,))
    pred = cv2.imread(os.path.join(root, name + '.png'), 0).astype(np.int32)
    pred_color = VOC.palette[pred.ravel()].reshape(pred.shape+(3,))
    sp = cv2.imread(os.path.join(sp_root, name+'.png')).astype(np.int32)
    sp = sp[..., 0] + sp[..., 1]* 256 + sp[..., 2]*65536
    sp_color = draw_sp(sp, img)
    score = np.load(os.path.join(root, name + '_score.npy')).astype(np.float32)
    entropy = np.load(os.path.join(root, name + '_entropy.npy')).astype(np.float32)

    score = score / score.max()
    score = get_score_map(score, img)
    entropy = get_score_map(entropy, img)

    demo = imhstack([img, gt_color, sp_color, pred_color, score, entropy], 360)
    return demo

def imshow(image):
    _CV = True
    if _CV:
        cv2.imshow("demo", image)
        k = cv2.waitKey(0)
        while k != 27 :
            k = cv2.waitKey(0)
    else:
        plt.imshow(image[..., ::-1])
        plt.show()

def show_some_samples():
    demos = []
    for name in names[:8]:
        demo = sample(name)
        demos.append(demo)

    demos = imvstack(demos)
    imshow(demos)

    #cv2.imwrite('/home/junsong/Downloads/demo.jpg', demos)

show_some_samples()

