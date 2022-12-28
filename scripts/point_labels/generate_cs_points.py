import sys
sys.path.append('../../')
from core.utils import *
import multiprocessing as mp
import json


def sample_ins_points(name):
    city, id0, id1 = name.split('_')
    prefix = os.path.basename(cs_gt_root)

    # train id
    srcLblId = os.path.join(cs_gt_root, subset, city, name + '_' + prefix + '_labelIds.png')
    assert os.path.exists(srcLblId), srcLblId
    lblId = cv2.imread(srcLblId, 0)
    trainId = CS.id2trainId[lblId.ravel()].reshape(lblId.shape)
    h, w = trainId.shape

    # instances
    srcPoly = os.path.join(cs_gt_root, subset, city, name + '_' + prefix + '_polygons.json')
    assert os.path.exists(srcPoly), srcPoly
    with open(srcPoly) as f:
        poly = json.load(f)

    # sampling
    results = []
    for obj in poly['objects']:
        class_name = obj['label']
        if class_name.endswith('group'):
            class_name = class_name[:-len('group')]
        L = CS.name2trainId[class_name]
        if L == 255: continue

        pts = [np.array(obj['polygon']).reshape(-1, 1, 2)]
        polygon_mask = cv2.fillPoly(np.zeros((h, w, 3), np.uint8), pts, (255, 255, 255))[..., 0]

        cand_mask = (polygon_mask > 0) & (trainId == L)
        cand_h, cand_w = np.nonzero(cand_mask)

        if len(cand_h) > 0:
            idx = np.random.randint(0, cand_h.size)
            results.append([L, cand_h[idx], cand_w[idx]])

    res_str = ' '.join([name] + [','.join([str(val) for val in res]) for res in results])
    return res_str

def sample_cls_points_N_times(name, N):
    city, id0, id1 = name.split('_')
    prefix = os.path.basename(cs_gt_root)

    # trainid
    srcLblId = os.path.join(cs_gt_root, subset, city, name + '_' + prefix + '_labelIds.png')
    assert os.path.exists(srcLblId), srcLblId
    lblId = cv2.imread(srcLblId, 0)
    trainId = CS.id2trainId[lblId.ravel()].reshape(lblId.shape)
    h, w = trainId.shape

    # sampling
    results = []
    for L in np.unique(trainId):
        if L == 255: continue

        cand_mask = trainId == L
        cand_h, cand_w = np.nonzero(cand_mask)

        if len(cand_h) > 0:
            prev_idx = []
            for i in range(N):
                idx = np.random.randint(0, cand_h.size)
                while idx in prev_idx:
                    idx = np.random.randint(0, cand_h.size)
                prev_idx.append(idx)
                results.append([L, cand_h[idx], cand_w[idx]])

    res_str = ' '.join([name] + [','.join([str(val) for val in res]) for res in results])
    return res_str

def sample_cls_points(name):
    return sample_cls_points_N_times(name, 1)

def sample_cls_points_two(name):
    return sample_cls_points_N_times(name, 2)

def sample_cls_points_three(name):
    return sample_cls_points_N_times(name, 3)

def sample_points(method):
    root = os.path.join(cs_gt_root, subset)
    cities = os.listdir(root)
    tasks = []
    for city in cities:
        dirname = os.path.join(root, city)
        names = list(set(['_'.join(x.split('_')[:3]) for x in os.listdir(dirname)]))
        tasks += names
    print('Total: {}'.format(len(tasks)))

    if method == 'instance':
        sample_fn = sample_ins_points
    elif method == 'class':
        sample_fn = sample_cls_points
    else:
        raise RuntimeError(method)

    pool = mp.Pool(54)
    jobs = [pool.apply_async(sample_fn, (task,)) for task in tasks]
    result = [job.get() for job in jobs]

    if not os.path.exists(os.path.dirname(savefile)):
        os.makedirs(os.path.dirname(savefile))

    with open(savefile, 'w') as f:
        f.write('\n'.join(result))


if __name__ == '__main__':
    # cs_gt_root = '/home/junsong_fan/diskf/data/cityscape/gtFine'
    # subset = 'train'
    # savefile = './Cityscape_ins_point_uniform.txt'
    # sample_points('instance')

    cs_gt_root = '/home/junsong_fan/diskf/data/cityscape/gtCoarse'
    subset = 'train_extra'
    savefile = './Cityscape_cls_point_uniform_train_extra.txt'
    sample_points('class')


