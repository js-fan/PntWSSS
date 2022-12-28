import numpy as np
import cv2
import multiprocessing as mp
import pydensecrf.densecrf as dcrf
import os
from core.utils import *
from compute_iou import compute_cs_iou
from pathlib import Path

def refine_and_save_cs(img_src, src, dst, is_trainId=False):
    pos_prob = 0.9
    neg_prob = (1 - pos_prob) / 18

    img = cv2.imread(img_src)
    h, w = img.shape[:2]

    if is_trainId:
        pred = cv2.imread(src, 0).ravel()
        assert pred.max() <= 19, np.unique(pred)
    else:
        pred = CS.id2trainId[cv2.imread(src, 0).ravel()]
    assert pred.size == h * w, (img.shape, pred.shape)

    probs = np.full((19, h * w), neg_prob, np.float32)
    probs[pred, np.arange(h * w)] = pos_prob
    u = - np.log(probs + 1e-5)

    d = dcrf.DenseCRF2D(w, h, probs.shape[0])
    d.setUnaryEnergy(u)
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=img, compat=10)

    probs_crf = d.inference(20)
    probs_crf = np.array(probs_crf).reshape(19, h, w)
    if is_trainId:
        pred_crf = probs_crf.argmax(axis=0).reshape(h, w).astype(np.uint8)
    else:
        pred_crf = CS.trainId2id[probs_crf.argmax(axis=0).ravel()].reshape(h, w).astype(np.uint8)
    imwrite(dst, pred_crf)

def refine_cityscape(image_root, pred_root, save_root, image_suffix="_leftImg8bit.png", label_suffix=".png", num_threads=16):
    predictions = [x.resolve() for x in Path(pred_root).rglob(f"*{label_suffix}")]
    images = [x.resolve() for x in Path(image_root).rglob(f"*{image_suffix}")]
    assert len(images) == len(predictions), (len(images), len(predictions))
    predictions = sorted(predictions, key=lambda x: x.name)
    images = sorted(images, key=lambda x: x.name)

    tasks = []
    for img, pred in zip(images, predictions):
        pred_crf = (Path(save_root) / pred.name).resolve()
        tasks.append((str(img), str(pred), str(pred_crf), True))

    pool = mp.Pool(num_threads)
    jobs = [pool.apply_async(refine_and_save_cs, task) for task in tasks]
    [job.get() for job in jobs]

def DEP_refine_cityscape(image_root, pred_root, save_root, suffix='_leftImg8bit.png', num_threads=16):
    cities = os.listdir(pred_root)
    tasks = []
    for city in cities:
        srcs = os.listdir(os.path.join(pred_root, city))

        for src in srcs:
            prefix = '_'.join(src.split('_')[:3])
            image = os.path.join(image_root, city, prefix + suffix)
            pred = os.path.join(pred_root, city, src)
            pred_crf = os.path.join(save_root, city, src)
            assert os.path.exists(image), image
            assert os.path.exists(pred), pred
            tasks.append((image, pred, pred_crf))
    
    pool = mp.Pool(num_threads)
    jobs = [pool.apply_async(refine_and_save_cs, task) for task in tasks]
    [job.get() for job in jobs]


if __name__ == '__main__':
    image_root = '/home/junsong_fan/diskf/data/cityscape/leftImg8bit/val'
    image_root = './data/cityscapes/leftImg8bit/val'
    gt_root = './data/cityscapes/gtFine/val'
    #pred_root = './snapshot_train_features/vgg16_largefov_bs16_lr0.0025_ep40_p1n64k2048t0.07s1_(predict_20_5)_mlp_l2_mask/results/prediction_direct/val'
    #save_root = './snapshot_train_features/vgg16_largefov_bs16_lr0.0025_ep40_p1n64k2048t0.07s1_(predict_20_5)_mlp_l2_mask/results/prediction_crf/val'

    dirs = [
        #'./snapshot/abl2/cityscape/vgg16_largefov_Ep40_Bs16_Lr0.001_GPU4_Size513x1025_mlp_fc7_T0.1_spTh0.1_capQ64_img_lmCE1_lmSPCE1_lmMEM0.1_wPosAnti0.0/results/prediction',
        #'./snapshot/abl2/cityscape/vgg16_largefov_Ep40_Bs16_Lr0.001_GPU4_Size513x1025_mlp_fc7_T0.1_spTh0.1_capQ64_sp_lmCE1_lmSPCE1_lmMEM0.0/results/prediction',
        #'./snapshot/ablation/cityscape/point_vgg16_largefov_Ep40_Bs16_Lr0.001_GPU4_Size513x1025_mlp_fc7_T0.1_spTh0.1_capQ64_sp_lmCE1_lmSPCE0.0_lmMEM0.0_noSEP_baselineCE/results/prediction',
        # './snapshot/abl2/cityscape/vgg16_largefov_Ep40_Bs16_Lr0.001_GPU4_Size513x1025_mlp_fc7_T0.1_spTh0.1_capQ64_img_lmCE1_lmSPCE1_lmMEM0.1_wPosAnti0.0/results/prediction_test',
        #'./work_dir4_redsum/pnt_size10_pl0.9/vgg16_largefov/results'

        #'./snapshots/cs_pmask_cl_lambda0.1/ProjectionModel/results',
        './snapshots/cs_pmask_cl_lambda0/ProjectionModel/results',
        #'./snapshots/cs_naive_base/ProjectionModel/results',
        # cs_pmask_cl
        # cs_pmask_cl_lambda0
        # cs_pmask_cl_lambda0.2
        # cs_pmask_cl_lambda0.5
        # cs_pmask_cl_lambda0.5_mem16
        # cs_pmask_cl_lambda0.5_mem256
        # cs_pmask_cl_lambda0.5_mem4
        # cs_pmask_cl_lambda0.5_mem8
        # cs_pmask_cl_lambda0.7
        # cs_pmask_cl_mem8
        # cs_pmask_cl_mem8_warm8_th5
        # cs_pmask_cl_mem8_warm8_th7
        # cs_square513_inbatch
        # cs_square513_lambda0
        # cs_square513_mem16_th03_warm8
        # cs_square513_mem16_th03_warm8_lm0.1
        # cs_square513_mem16_th03_warm8_lm0.2
        # cs_square513_mem512_th03_warm8_lm0.1
        # cs_square513_mem64_th03_warm10_lm0.1_norm
        # cs_square513_mem64_th03_warm8_lm0.1
        # cs_square513_mem64_th03_warm8_lm0.2
    ]

    a = [    # './snapshot_train_features/size1024_vgg16_largefov_bs16_lr0.0025_ep40_p1n2048k3072t0.07s0_(th-predict-soft_20_5_0.5)_linear_l2_mask_noCtLoss_k3x3_newCtLossVerify_randPriority',
        # './snapshot_train_features/size1024_vgg16_largefov_bs16_lr0.0025_ep40_p1n4096k8192t0.07s0.01_(th-predict-soft_20_5_0.5)_linear_l2_mask_k3x3_newCtLossVerify_randPriority',
        # './snapshot_train_features/size1024_vgg16_largefov_bs16_lr0.0025_ep40_p1n8192k9216t0.07s0.01_(th-predict-soft_20_5_0.5)_linear_l2_mask_k3x3_newCtLossVerify_randPriority',
        # './snapshot_train_features/size1024_vgg16_largefov_bs16_lr0.0025_ep40_p1n8192k9216t0.07s0.01_(th-predict-soft_20_5_0.5)_linear_l2_mask_k3x3_newCtLossVerify_randPriority_InitUpdate',

        # './snapshot_train_features/size1024_vgg16_largefov_bs16_lr0.0025_ep40_p1n2048k4096t0.07s0.01_(th-predict-soft_20_5_0.5)_linear_l2_mask_k3x3_newCtLossVerify',
        # './snapshot_train_features/size1024_vgg16_largefov_bs16_lr0.0025_ep40_p1n512k2048t0.07s0.01_(th-predict-soft_20_5_0.5)_linear_l2_mask_k3x3_newCtLossVerify',
        # './snapshot_train_features/size1024_vgg16_largefov_bs16_lr0.0025_ep40_p1n512k2048t0.07s0.1_(th-predict-soft_20_5_0.5)_linear_l2_mask_k3x3_newCtLossVerify',
        # './snapshot_train_features/size1024_vgg16_largefov_bs16_lr0.0025_ep40_p1n64k2048t0.07s0.01_(th-predict-soft_20_5_0.5)_linear_l2_mask_k3x3_newCtLossVerify',
        # './snapshot_clear/cityscape/size1024_resnet101_largefov_bs16_lr0.0025_ep40_lrMult1_CXX',
    ]

    #for dirname in dirs:
    #    pred_root = os.path.join(dirname, 'results', 'prediction_direct', 'val')
    #    save_root = os.path.join(dirname, 'results', 'prediction_crf', 'val')

    for pred_root in dirs:
        save_root = pred_root + '_crf'
        print(pred_root)
        image_root_ = image_root[:-3] + 'test' if pred_root.endswith('test') else image_root
        refine_cityscape(image_root_, pred_root, save_root, num_threads=32)

        if not pred_root.endswith('test'):
            iou_cls, iou_cat = compute_cs_iou(save_root, gt_root)
            msg_cls = '\t'.join(['%.3f' % x for x in iou_cls])
            print(iou_cls.mean())
            print(msg_cls)
            with open('./cs_crf_results.txt', 'a') as f:
                f.write(save_root + '\n' + msg_cls + '\n\n')

