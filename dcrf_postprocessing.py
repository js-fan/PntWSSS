import pydensecrf.densecrf as dcrf
import numpy as np
import multiprocessing as mp
import os
from compute_iou import *

NUM_CLS = 21
def process_seed_full_class(src, img_src, dst, confidence=0.5, unsure_threshold=0.9, ignore_index=255):
    seed = cv2.imread(src, 0)
    unique = np.array([x for x in np.unique(seed) if x != ignore_index])
    if len(unique) <= 1:
        cv2.imwrite(dst, seed)
        return

    img = cv2.imread(img_src)
    assert seed.shape == img.shape[:2], (seed.shape, img.shape)
    h, w = seed.shape
    n = NUM_CLS

    neg_conf = (1 - confidence) / (n - 1)
    prob = np.full((ignore_index+1, h * w), neg_conf, np.float32)
    prob[seed.ravel(), np.arange(h * w)] = confidence
    prob = prob[:n] / prob[:n].sum(axis=0, keepdims=True)

    u = - np.log(np.maximum(prob, 1e-5))

    d = dcrf.DenseCRF2D(w, h, n)
    d.setUnaryEnergy(u)

    # vgg
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=img, compat=10)

    # resnet
    d.addPairwiseGaussian(sxy=1, compat=3)
    d.addPairwiseBilateral(sxy=67, srgb=3, rgbim=img, compat=4)

    out = d.inference(10)
    out = np.array(out).reshape(n, h, w)
    out_seed = out.argmax(axis=0)

    #unsure = out.max(axis=0) < unsure_threshold
    #out_seed[unsure] = ignore_index
    out_seed = out_seed.reshape(h, w).astype(np.uint8)

    cv2.imwrite(dst, out_seed)

def process_seed(src, img_src, dst, confidence=0.7, unsure_threshold=0.9, ignore_index=255):
    seed = cv2.imread(src, 0)
    unique = np.array([x for x in np.unique(seed) if x != ignore_index])
    if len(unique) <= 1:
        cv2.imwrite(dst, seed)
        return

    img = cv2.imread(img_src)
    assert seed.shape == img.shape[:2], (seed.shape, img.shape)
    h, w = seed.shape
    n = len(unique)

    label2id = np.arange(ignore_index+1)
    label2id[unique] = np.arange(n)
    id2label = np.arange(ignore_index+1)
    id2label[np.arange(n)] = unique

    neg_conf = (1 - confidence) / (n - 1)
    prob = np.full((ignore_index+1, h * w), neg_conf, np.float32)
    prob[label2id[seed.ravel()], np.arange(h * w)] = confidence
    prob = prob[:n] / prob[:n].sum(axis=0, keepdims=True)

    u = - np.log(prob)

    d = dcrf.DenseCRF2D(w, h, n)
    d.setUnaryEnergy(u)
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=img, compat=10)

    out = d.inference(10)
    out = np.array(out).reshape(n, h, w)
    out_seed = out.argmax(axis=0)

    unsure = out.max(axis=0) < unsure_threshold
    out_seed[unsure] = ignore_index
    out_seed = id2label[out_seed.ravel()].reshape(h, w).astype(np.uint8)

    cv2.imwrite(dst, out_seed)

def process_voc(src_root, dst_root=None, fn=None, compute_iou='val'):
    img_root = '/home/junsong_fan/diskf/data/VOC2012/JPEGImages'
    dst_root = (src_root + '_crf') if dst_root is None else dst_root
    if not os.path.exists(dst_root):
        os.makedirs(dst_root)

    names = os.listdir(src_root)
    tasks = [[  os.path.join(src_root, name),
                os.path.join(img_root, name[:-3] + 'jpg'),
                os.path.join(dst_root, name),
             ] for name in names]
    pool = mp.Pool(48)

    fn = process_seed_full_class if fn is None else fn
    jobs = [pool.apply_async(fn, task) for task in tasks]
    [job.get() for job in jobs]

    if compute_iou:
        gt_root = '/home/junsong_fan/diskf/data/VOC2012/extra/SegmentationClassAug'
        compute_voc_iou(dst_root, gt_root, compute_iou)

if __name__ == '__main__':
    #process_voc('./snapshot/region_mem/cian_seed_vgg16_largefov_Ep20_Bs16_Lr0.001_GPU4_ps16_th0.3_maxinter/results/prediction')
    #process_voc('./snapshot/region_mem/cian_seed_res50_largefov_Ep20_Bs16_Lr0.001_GPU4_ps16_th0.3_maxinter/results/prediction')
    #process_voc('./snapshot/segmentation/cian_seed_res50_largefov_Ep20_Bs16_Lr0.001_GPU4/results/prediction')
    #process_voc('./snapshot/refpool/cian_seed_self_training_vgg16_largefov_Ep20_Bs16_Lr0.001_GPU4_ps8_topk0.01_th0.0_selfpred/results/prediction')
    #process_voc('./snapshot/pre_ablation/voc/point_vgg16_largefov_Ep20_Bs16_Lr0.0005_GPU4_Size321x321_mlp_fc7_l2m3.8_ce1_spth0.1_spce1_spfe0_incl0_mem0.0_trueSpLabel/results/prediction', fn=process_seed_full_class)

    #process_voc('./snapshot/pre_ablation/voc/point_vgg16_largefov_Ep20_Bs16_Lr0.0005_GPU4_Size321x321_mlp_fc7_l2m3.8_MEM64_batchwise_largest_ce1_spth0.1_spce1_spfe0_incl0_mem1_trueSpLabel/results/prediction', fn=process_seed_full_class)
    #process_voc('./snapshot/pre_ablation/voc/point_vgg16_largefov_Ep20_Bs16_Lr0.0005_GPU4_Size321x321_mlp_fc7_l2m3.8_MEM64_batchwise_largest_ce1_spth0.1_spce1_spfe0_incl0_mem1_icdSp/results/prediction', fn=process_seed_full_class)
    #process_voc('./snapshot/ablation/voc/point_vgg16_largefov_Ep20_Bs16_Lr0.0005_GPU4_Size321x321_mlp_fc7_T0.1_spTh0.1_capQ64_pix_lmCE1_lmSPCE1_lmMEM1_clsLblFilter/results/prediction', fn=process_seed_full_class)
    #process_voc('./snapshot/ablation/voc/point_vgg16_largefov_Ep20_Bs16_Lr0.0005_GPU4_Size321x321_mlp_fc7_T0.1_spTh0.1_capQ16_sp+img_lmCE1_lmSPCE1_lmMEM1_sepSum/results/prediction', fn=process_seed_full_class)

    #process_voc('./snapshot/ablation/voc/point_vgg16_largefov_Ep20_Bs16_Lr0.001_GPU4_Size321x321_mlp_fc7_T0.1_spTh0.1_capQ64_img_lmCE1_lmSPCE1_lmMEM1_lr1E-3/results/prediction', fn=process_seed_full_class)
    process_voc('./snapshot/ablation/voc/point_vgg16_largefov_Ep20_Bs16_Lr0.001_GPU4_Size321x321_mlp_fc7_T0.1_spTh0.1_capQ64_img_lmCE1_lmSPCE1_lmMEM1_lr1E-3/results/prediction_test', fn=process_seed_full_class, compute_iou=None)

    #process_voc('./snapshot/abl3/voc/point_deeplab_v2_Ep20_Bs16_Lr0.00025_GPU4_Size321x321_mlp_fc7_T0.1_spTh0.1_capQ64_img_lmCE1_lmSPCE1_lmMEM1/results/prediction', fn=process_seed_full_class)

    process_voc('./snapshot/abl3/voc/point_deeplab_v2_Ep20_Bs16_Lr0.00025_GPU4_Size321x321_mlp_fc7_T0.1_spTh0.1_capQ64_img_lmCE1_lmSPCE1_lmMEM1/results/prediction_test', fn=process_seed_full_class, compute_iou=None)


