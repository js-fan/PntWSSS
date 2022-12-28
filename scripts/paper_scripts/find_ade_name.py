import sys
sys.path.append('../../')
from core.utils import *
import multiprocessing as mp


ade_root = '/home/junsong_fan/diskf/data/ADE20k/validation'
candidates = [os.path.join(ade_root, x) for x in os.listdir(ade_root) if x.endswith('.jpg')]

def compute_dist(src, dst):
    target = cv2.imread(src)
    h, w = target.shape[:2]

    cand = cv2.imread(dst)
    h2, w2 = cand.shape[:2]

    if h > w:
        ratio = float(h) / w
        r2 = float(h2) / w2
    else:
        ratio = float(w) / h
        r2 = float(w2) / h2

    #if np.abs(ratio - r2) > 0.2:
    #    return dst, np.inf

    cand = cv2.resize(cand, (w, h))
    dist = np.square(target.astype(np.float32) - cand.astype(np.float32)).sum()
    return dst, dist

def compute_all_dist(src):
    pool = mp.Pool(54)
    jobs = [pool.apply_async(compute_dist, (src, dst)) for dst in candidates]
    results = [job.get() for job in jobs]
    names = [res[0] for res in results]
    dist = np.array([res[1] for res in results])

    topk = dist.argsort()[:10]
    images = []
    for idx in topk:
        print(names[idx], dist[idx])
        images.append(cv2.imread(names[idx]))

    imwrite('./candidates.jpg', imvstack(images))

def copy_results():
    for src in res:
        name = os.path.basename(src)

        for i, pred_root in enumerate(preds):
            pred_src = os.path.join(pred_root, name[:-3] + 'png')
            pred = cv2.imread(pred_src, 0)
            pred = ADE.palette[pred.ravel()].reshape(pred.shape+(3,))

            imwrite('./ade_results/{}_{}.jpg'.format(i, name[:-4]), pred)


if __name__ == '__main__':
    #compute_all_dist('./ade.demo.2.png')

    preds = [
            '../../snapshot/ablation/ade20k/point_vgg16_largefov_Ep40_Bs16_Lr0.001_GPU4_Size321x321_mlp_fc7_T0.1_spTh0.1_capQ64_img_lmCE1_lmSPCE0.1_lmMEM1.0_rampup35/results/prediction',
            '../../snapshot/ablation/ade20k/point_deeplab_v2_Ep40_Bs16_Lr0.00025_GPU4_Size321x321_mlp_fc7_T0.1_spTh0.1_capQ64_img_lmCE1_lmSPCE0.1_lmMEM1.0/results/prediction',
    ]

    res = [
            '/home/junsong_fan/diskf/data/ADE20k/validation/ADE_val_00000156.jpg',
            '/home/junsong_fan/diskf/data/ADE20k/validation/ADE_val_00001789.jpg'
    ]

    copy_results()


