from numpy import expm1
from core.utils import *
from core.data import PointAugDatasetWithSP
import yaml
from easydict import EasyDict

SPPALETTE = np.random.randint(0, 256, (1024, 3), np.uint8)

def check_cs():
    Path(saveDir).mkdir(parents=True, exist_ok=True)
    with open('./configs/base_cs.yaml') as f:
        cfg = EasyDict(yaml.safe_load(f))

    cfg.data.train.image_root = [
        cfg.data.train.image_root,
        '/home/junsong_fan/diskf/data/cityscapes/gtFine/train'
    ]
    cfg.data.train.rand_crop = False
    cfg.data.train.rand_short = [512, 512]
    dataset = PointAugDatasetWithSP(image_suffix=[".png", "_color.png", ".png"], **cfg.data.train)
    indices = range(0, min(len(dataset), 100), 10)
    examples = []
    for i in indices:
        image, gt, sp, pnt_map = dataset[i]

        image = PointAugDatasetWithSP.denormalize(image)
        gt = PointAugDatasetWithSP.denormalize(gt)
        sp = SPPALETTE[sp.data.cpu().numpy().ravel() % 1024].reshape(sp.shape+(3,))
        pnt_map = CS.palette[pnt_map.data.cpu().numpy().ravel()].reshape(pnt_map.shape+(3,))
        examples.append(imhstack([image, gt, sp, pnt_map], height=360))
    examples = imvstack(examples) 
    cv2.imwrite(str(Path(saveDir) / "cs_nocrop.jpg"), examples)

    dataset.rand_crop = True
    dataset.rand_short = [512, 768]
    examples = []
    for i in indices:
        image, gt, sp, pnt_map = dataset[i]

        image = PointAugDatasetWithSP.denormalize(image)
        gt = PointAugDatasetWithSP.denormalize(gt)
        sp = SPPALETTE[sp.data.cpu().numpy().ravel() % 1024].reshape(sp.shape+(3,))
        pnt_map = CS.palette[pnt_map.data.cpu().numpy().ravel()].reshape(pnt_map.shape+(3,))
        examples.append(imhstack([image, gt, sp, pnt_map], height=360))
    examples = imvstack(examples) 
    cv2.imwrite(str(Path(saveDir) / "cs_crop.jpg"), examples)


if __name__ == '__main__':
    saveDir = './cache'
    check_cs()