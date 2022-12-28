import numpy as np
import cv2
import multiprocessing as mp
import os
from core.utils import *

def compute_ade_pacc(pred_root, gt_root, data_list=None, logger=None):
    gt_dict = {x.rsplit('.', 1)[0]: os.path.join(gt_root, x) for x in os.listdir(gt_root) if x.endswith('.png')}
    pred_dict = {x.rsplit('.', 1)[0]: os.path.join(pred_root, x) for x in os.listdir(pred_root) if x.endswith('.png')}

    if data_list is not None:
        with open(data_list) as f:
            names = [x.strip().split(' ')[0] for x in f.readlines()]
    else:
        names = list(gt_dict.keys())

    for name in names:
        assert name in gt_dict, 'Cannot find gt: {}'.format(name)
        assert name in pred_dict, 'Cannot find pred: {}'.format(name)
        assert os.path.exists(gt_dict[name]), 'Cannot find gt: {}'.format(gt_dict[name])
        assert os.path.exists(pred_dict[name]), 'Cannot find pred: {}'.format(pred_dict[name])

    pairs = [[gt_dict[name], pred_dict[name]] for name in names]
    info(logger, 'In total {} samples'.format(len(pairs)))

    # pixel acc
    pacc = compute_pixel_acc(pairs, 150, num_threads=16)
    info(logger, 'ADE20k pixel acc: {}\n{}'.format(pacc.mean(), pacc))
    return pacc

def _compute_thread(gt_src, pred_src, num_classes, map_func, method):
    map_func = np.arange(max(num_classes, 256), dtype=np.int64) if map_func is None else map_func
    gt = cv2.imread(gt_src, 0)
    pred = cv2.imread(pred_src, 0)
    if gt.shape != pred.shape:
        pred = cv2.resize(pred, gt.shape[::-1], interpolation=cv2.INTER_NEAREST)
    
    gt = map_func[gt.ravel()]
    pred = map_func[pred.ravel()]
    valid = (gt < num_classes) & (pred < num_classes)
    gt = gt[valid]
    pred = pred[valid]
    labels = [L for L in np.unique(gt) if L < num_classes]

    outputs = np.zeros((num_classes, 2), np.int64)
    for L in labels:
        gt_ = gt == L
        pred_ = pred == L

        if method == 1:
            outputs[L][0] += (~(gt_ ^ pred_)).sum()
            outputs[L][1] += gt_.size
        else:
            outputs[L][0] +=  (gt_ & pred_).sum()
            outputs[L][1] += gt_.sum()
    return outputs

def compute_pixel_acc(pairs, num_classes, map_func=None, num_threads=16, method=3):
    pool = mp.Pool(num_threads)
    jobs = [pool.apply_async(_compute_thread, (gt_src, pred_src, num_classes, map_func, method)) for gt_src, pred_src in pairs]
    outputs = [job.get() for job in jobs]

    outputs = np.array(outputs).sum(axis=0)
    if method in [1, 2]:
        pixel_acc = outputs[:, 0] / np.maximum(outputs[:, 1], 1)
    elif method in [3]:
        all_cls_sum = outputs.sum(axis=0)
        pixel_acc = all_cls_sum[0] / np.maximum(all_cls_sum[1], 1)
    return pixel_acc

if __name__ == '__main__':
    for dirname in [
            './snapshot/ablation/ade20k/point_vgg16_largefov_Ep40_Bs16_Lr0.001_GPU4_Size321x321_mlp_fc7_T0.1_spTh0.1_capQ64_sp_lmCE1_lmSPCE0.0_lmMEM0.0/results/prediction',
            './snapshot/ablation/ade20k/point_vgg16_largefov_Ep40_Bs16_Lr0.001_GPU4_Size321x321_mlp_fc7_T0.1_spTh0.1_capQ64_img_lmCE1_lmSPCE0.1_lmMEM0.0_rampup35/results/prediction',
            './snapshot/ablation/ade20k/point_vgg16_largefov_Ep40_Bs16_Lr0.001_GPU4_Size321x321_mlp_fc7_T0.1_spTh0.1_capQ64_img_lmCE1_lmSPCE0.1_lmMEM1.0_rampup35/results/prediction',
            './snapshot/ablation/ade20k/point_deeplab_v2_Ep40_Bs16_Lr0.00025_GPU4_Size321x321_mlp_fc7_T0.1_spTh0.1_capQ64_img_lmCE1_lmSPCE0.0_lmMEM0.0/results/prediction',
            './snapshot/ablation/ade20k/point_deeplab_v2_Ep40_Bs16_Lr0.00025_GPU4_Size321x321_mlp_fc7_T0.1_spTh0.1_capQ64_img_lmCE1_lmSPCE0.1_lmMEM0.0/results/prediction',
            './snapshot/ablation/ade20k/point_deeplab_v2_Ep40_Bs16_Lr0.00025_GPU4_Size321x321_mlp_fc7_T0.1_spTh0.1_capQ64_img_lmCE1_lmSPCE0.1_lmMEM1.0/results/prediction'
    ]:
        print(dirname)
        compute_ade_pacc(dirname,
            '/home/junsong_fan/diskf/data/ADE20k/ADE_validation_annotations')
        print('')

    # compute_ade_pacc(
    #         #'../snapshot_clear/ade20k/baseline/vgg16_largefov_bs16_lr0.0025_ep40_lrMult1/results/prediction_direct/val',
    #         #'../snapshot_clear/ade20k/outbatch_contrast/vgg16_largefov_bs16_lr0.0025_ep40_lrMult1_(linear_l2_predSoft_t0.5_s1)_(p1_n2048_k3072_t0.07_gs0.01)_CXX/results/prediction_direct/val',
    #         '../snapshot_clear/ade20k/baseline/resnet101_largefov_bs16_lr0.0025_ep40_lrMult1_CXX/results/prediction_direct/val',
    #         '/home/junsong_fan/diskf/data/ADE20k/ADE_validation_annotations'
    # )
