import sys 
sys.path.append("../../")
from core.utils import *
import json

np.random.seed(0)
palette = np.random.randint(0, 256, (1000, 3), np.uint8)

cs_gt_root = '/home/junsong_fan/diskf/data/cityscape/gtFine/val/frankfurt'
names = list(set(['_'.join(x.split('_')[:3]) for x in  os.listdir(cs_gt_root)]))
names.sort()

examples = []
for name in names[:10]:
    gtColor = os.path.join(cs_gt_root, name + '_gtFine_color.png')
    gtInsId = os.path.join(cs_gt_root, name + '_gtFine_instanceIds.png')
    gtLblId = os.path.join(cs_gt_root, name + '_gtFine_labelIds.png')
    gtPoly = os.path.join(cs_gt_root, name + '_gtFine_polygons.json')

    assert os.path.exists(gtColor), gtColor
    assert os.path.exists(gtInsId), gtInsId
    assert os.path.exists(gtLblId), gtLblId
    assert os.path.exists(gtPoly), gtPoly

    imgs = [cv2.imread(gtColor)]

    insId = cv2.imread(gtInsId, 0)
    imgs.append(palette[insId.ravel()].reshape(insId.shape+(3,)))

    lblId = cv2.imread(gtLblId, 0)
    imgs.append(palette[lblId.ravel()].reshape(lblId.shape+(3,)))

    # polygon
    with open(gtPoly) as f:
        poly = json.load(f)

    imgPoly = imgs[1].copy()
    for obj in poly['objects']:
        class_name = obj['label']
        polygon = obj['polygon']

        if class_name.endswith('group'):
            class_name = class_name[:-len('group')]

        L = CS.name2id[class_name]
        color = [int(c) for c in CS.paletteId[L]]

        pts = [np.int32(polygon).reshape(-1, 1, 2)]
        #cv2.fillPoly(imgPoly, pts, color)
        cv2.polylines(imgPoly, pts, True, color, 5)

    imgs.append(imgPoly)


    print(name, len(np.unique(insId)), np.unique(insId))

    examples.append(imhstack(imgs, height=360))

imwrite('./ins_demo.jpg', imvstack(examples))
