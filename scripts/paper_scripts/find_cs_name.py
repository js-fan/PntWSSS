import sys
sys.path.append('../../')
from core.utils import *
import multiprocessing as mp

cs_root = '/home/junsong_fan/diskf/data/cityscape/leftImg8bit_jpg/val'
candidates = []
for city in os.listdir(cs_root):
    candidates += [os.path.join(cs_root, city, src) for src in os.listdir(os.path.join(cs_root, city))]


def compute_dist(src, dst):
    target = cv2.imread(src)
    h, w = target.shape[:2]

    cand = cv2.resize(cv2.imread(dst), (w, h))
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
    for src in found:
        name = os.path.basename(src)
        city, id0, id1 = os.path.basename(name).split('_')[:3]

        for i, pred_root in enumerate(preds):
            pred_src = os.path.join(pred_root, city, '{}_{}_{}_pred_labelIds.png'.format(city, id0, id1))
            pred = cv2.imread(pred_src, 0)
            pred = CS.paletteId[pred.ravel()].reshape(pred.shape+(3,))

            imwrite('./cs_results/{}_{}.jpg'.format(i, name[:-4]), pred)

if __name__ == '__main__':
    to_find = [
            './visualization/1.png',
            './visualization/2.png',
            './visualization/3.png',
            './visualization/4.png',
    ]

    #compute_all_dist(to_find[3])

    found = [
            '/home/junsong_fan/diskf/data/cityscape/leftImg8bit_jpg/val/lindau/lindau_000021_000019_leftImg8bit.jpg',
            '/home/junsong_fan/diskf/data/cityscape/leftImg8bit_jpg/val/munster/munster_000091_000019_leftImg8bit.jpg',
            '/home/junsong_fan/diskf/data/cityscape/leftImg8bit_jpg/val/lindau/lindau_000004_000019_leftImg8bit.jpg',
            '/home/junsong_fan/diskf/data/cityscape/leftImg8bit_jpg/val/frankfurt/frankfurt_000001_062016_leftImg8bit.jpg',
    ]

    preds = [
        '../../snapshot/ablation/cityscape/point_vgg16_largefov_Ep40_Bs16_Lr0.001_GPU4_Size513x1025_mlp_fc7_T0.1_spTh0.1_capQ64_sp_lmCE1_lmSPCE0.0_lmMEM0.0_noSEP_baselineCE/results/prediction_crf',
        '../../snapshot/abl2/cityscape/vgg16_largefov_Ep40_Bs16_Lr0.001_GPU4_Size513x1025_mlp_fc7_T0.1_spTh0.1_capQ64_sp_lmCE1_lmSPCE1_lmMEM0.0/results/prediction_crf',
        '../../snapshot/abl2/cityscape/vgg16_largefov_Ep40_Bs16_Lr0.001_GPU4_Size513x1025_mlp_fc7_T0.1_spTh0.1_capQ64_img_lmCE1_lmSPCE1_lmMEM0.1_wPosAnti0.0/results/prediction_crf',
    ]

    copy_results()
