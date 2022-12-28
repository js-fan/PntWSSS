import cv2
import numpy as np
import json
import tempfile
from cityscapesscripts.helpers.labels import labels as _labels
from cityscapesscripts.preparation.json2instanceImg import json2instanceImg, createInstanceImage, Annotation
from scipy.stats import multivariate_normal
import multiprocessing as mp
from pathlib import Path

_names = [L.name for L in _labels]
_trainIds = [L.trainId for L in _labels]

def name2color(name):
    idx = _names.index(name)
    return _labels[idx].color

def trainId2color(trainId):
    idx = _trainIds.index(trainId)
    return _labels[idx].color

def show_img(img, name="demo"):
    cv2.imshow(name, img[..., ::-1])
    cv2.waitKey(1)

def get_ins_img(jsonfile):
    annotation = Annotation()
    annotation.fromJsonFile(jsonfile)
    ins_img = createInstanceImage(annotation, 'trainIds')
    return np.array(ins_img)

def get_2d_gaussian(coords, mean, cov):
    h, w = coords.shape[:2]
    return multivariate_normal.pdf(coords.reshape(h*w, 2), mean, cov).reshape(h, w)

def find_index(array, val, begin, end):
    idx = begin
    while array[idx] < val:
        idx += 1
    return idx

def sample_points(jsonfile, mode='uniform', gaussian_cov=3, debug=False):
    insImg = get_ins_img(jsonfile)
    h, w = insImg.shape

    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    coords = np.concatenate([X[..., np.newaxis], Y[..., np.newaxis]], -1)

    _vis_density = np.zeros((h, w), np.float32)

    res = []
    for idVal in np.unique(insImg):
        if (idVal == 255) or (idVal == -1):
            continue
        mask = insImg == idVal

        if mode == 'uniform':
            density = np.ones((h, w), np.float32)
        elif mode in ['gaussian_center', 'guassian_border']:
            numComp, compMap, compStat, compCenter = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 4, cv2.CV_8U)
            if (numComp == 2):
                compIdx = 1
            else:
                compPrior = compStat[1:, -1]
                compPrior = np.cumsum(compPrior / compPrior.sum())
                compIdx = find_index(compPrior, np.random.rand(), 0, compPrior.size-1) + 1

            mask = compMap == compIdx
            span = compStat[compIdx][2:4]
            center = compCenter[compIdx]
            density = get_2d_gaussian(coords, center, span * gaussian_cov)

            if mode == 'gaussian_border':
                density = density.max() - density
        else:
            raise RuntimeError("Unknown mode:" + mode)

        if debug:
            _vis_density = np.maximum(_vis_density, density * mask / density.max())

        density = density[mask]
        _coords = coords[mask]

        # i: cum[i-1] < rand <= cum[i]
        cum_density = np.cumsum(density)
        cum_density /= cum_density[-1]
        assert cum_density[-1] <= 1, cum_density[-1]
        randnum = np.random.rand()
        idx = find_index(cum_density, randnum, 0, cum_density.size-1)
        x, y = _coords[idx]

        trainId = idVal if idVal < 1000 else idVal // 1000
        res.append((trainId, x, y))
    if debug:
        return res, _vis_density
    return res

_MP = True
def prepare_annotation(root, dstFile, mode="uniform", **kwargs):
    suffix = '_polygons.json'
    #filenames = list(Path(root).glob("**/[!.]*" + suffix))
    filenames = list(Path(root).rglob(f"*{suffix}"))
    print(f"Processing {len(filenames)} files.")
    gaussian_cov = kwargs.get("gaussian_cov", 3)


    if _MP:
        pool = mp.Pool(64)
        jobs = [pool.apply_async(sample_points, (filename.resolve(), mode, gaussian_cov)) for filename in filenames]
        results = [job.get() for job in jobs]
    else:
        results = [sample_points(filename.resolve(), mode, gaussian_cov) for filename in filenames]

    output = []
    for filename, res in zip(filenames, results):
        prefix = filename.name.replace(suffix, '')
        labels = [f"{L},{x},{y}" for L, x, y in res]
        output.append(' '.join([prefix] + labels))

    Path(dstFile).parents[0].mkdir(parents=True, exist_ok=True)
    with open(dstFile, 'w') as f:
        f.write('\n'.join(output))

def test_code():
    res, density = sample_points('./aachen_000000_000019_gtFine_polygons.json', 'gaussian_center')
    img = cv2.imread('./aachen_000000_000019_gtFine_color.png')
    
    density = ((density / (density.max() + 1e-10)) * 255.99).astype(np.uint8)
    density = cv2.applyColorMap(density, cv2.COLORMAP_JET)
    
    for L, x, y in res:
        color = trainId2color(L)[::-1]
        cv2.drawMarker(img, (x, y), (0, 0, 0), markerType=cv2.MARKER_CROSS, thickness=10)
        cv2.drawMarker(img, (x, y), color, markerType=cv2.MARKER_CROSS, thickness=5)
    
    img = np.vstack([img, density])
    cv2.imshow("demo", img)
    cv2.waitKey(0)

if __name__ == '__main__':
    #prepare_annotation('./data/cityscapes/gtFine/train', './resources/labels/cityscapes/uniform_instance.txt', 'uniform')
    prepare_annotation('./data/cityscapes/gtFine/train', './resources/labels/cityscapes/center_instance.txt', 'gaussian_center')

