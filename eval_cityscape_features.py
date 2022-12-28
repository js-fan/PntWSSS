import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from core.data.Cityscape import CSPointDataset
from core.model.vgg import vgg16_largefov
from core.model.misc import *
from core.utils import * 
from compute_iou import compute_cs_iou
from collections import OrderedDict

import argparse
import os
import copy

def run_extract_feature(gpu, args, port):
    # env
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    world_size = len(args.gpus.split(','))
    is_distributed = world_size > 1

    if is_distributed:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = port
        dist.init_process_group('nccl', rank=gpu, world_size=world_size)
    is_log = gpu == 0

    # data
    dataset = CSPointDataset(args.test_image_root, None, args.test_gt_root, None, return_src=True)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False) if is_distributed else None
    loader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=False, drop_last=False,
            sampler=sampler, num_workers=1)

    # model
    mod = eval(args.model)(args.num_classes, dilation=12, upsample=args.upsample).to(gpu)
    mod.set_mode('seg_conv5')
    if args.force_pretrained:
        pretrained = args.force_pretrained
    else:
        pretrained = os.path.join(args.snapshot, 'checkpoint', '%s-%04d.pth' % (args.model, args.num_epochs-1))
    pretrained = torch.load(pretrained, map_location={'cuda:0': 'cuda:%s'%gpu})

    if args.force_pretrained:
        mod.init_params(pretrained, args.seed)

    #pretrained = OrderedDict()
    #for k, v in pretrained_.items():
    #    k = ('module.' + k) if not k.startswith('module.') else k
    #    pretrained[k] = v

    if is_distributed:
        mod = torch.nn.parallel.DistributedDataParallel(mod, device_ids=[gpu], find_unused_parameters=True)

    for k in pretrained.keys():
        if 'embedding' in k:
            del pretrained[k]
    if not args.force_pretrained:
        mod.load_state_dict(pretrained, strict=True)

    mod.train(False)

    get_save_name = lambda x: os.path.join(args.snapshot, 'cache', 'feature', '_'.join(os.path.basename(x).split('_')[:3]) + '.pkl')

    # test
    cnt = 0
    for image, label, src in loader:
        image = image.to(gpu)
        _, _, h, w = image.size()

        with torch.no_grad():
            logit, feature = mod(image)
            prob, pred = torch.softmax(logit, dim=1).max(dim=1)

            image = denormalize_image(image[0].data.cpu().numpy().transpose(1, 2, 0)[..., ::-1])
            prob = prob[0].data.cpu().numpy()
            pred = pred[0].data.cpu().numpy()
            label = label[0].data.cpu().numpy()
            feature = feature[0].data.cpu().numpy()

            pkldump(get_save_name(src[0]), (image, label, prob, pred, feature))
            cnt += 1

        if cnt >= 5:
            break

    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
def visualize_feature(filename, target_src):
    info(None, filename)
    image, label, prob, pred, feature = pklload(filename)
    label = CS.palette[label.ravel()].reshape(label.shape+(3,))
    prob = get_score_map(prob, image)
    pred = CS.palette[pred.ravel()].reshape(pred.shape+(3,))

    c, h, w = feature.shape
    feature_flat = feature.reshape(c, h * w).T
    #feature_flat -= feature_flat.mean(axis=1, keepdims=True)
    #feature_flat /= feature_flat.std(axis=1, keepdims=True)
    #feature_flat /= np.linalg.norm(feature_flat, axis=1, keepdims=True)

    #feat_embedded = TSNE(n_components=3).fit_transform(feature_flat)
    feat_embedded = PCA(n_components=3).fit_transform(feature_flat)
    #feat_embedded = feature_flat[:, np.random.permutation(c)[:3]]

    feat_embedded -= feat_embedded.min(axis=1, keepdims=True)
    feat_embedded /= np.maximum(feat_embedded.max(axis=1, keepdims=True), 1e-5)
    feat_embedded = (feat_embedded * 255).astype(np.uint8).reshape(h, w, 3)

    res = imhstack([image, label, prob, pred, feat_embedded], height=240)
    imwrite(target_src, res)
    return res

def run_visualize_features(args):
    #args.snapshot = os.path.join(args.snapshot, get_snapshot_name(args))
    filenames = filter(lambda x: x.endswith('.pkl'), os.listdir(os.path.join(args.snapshot, 'cache', 'feature')))
    pool = mp.Pool(32)
    jobs = [pool.apply_async(visualize_feature, (
        os.path.join(args.snapshot, 'cache', 'feature', name),
        os.path.join(args.snapshot, 'cache', 'result', name[:-3]+'jpg')
        ) ) for name in filenames]
    res = [job.get() for job in jobs]
    res = imvstack(res[:10])
    imwrite(os.path.join(args.snapshot, 'cache', 'vs_feat.jpg'), res)

def run(args, mode):
    args = copy.copy(args)
    #args.snapshot = os.path.join(args.snapshot, get_snapshot_name(args))
    args.seed = np.random.randint(0, 10000)
    world_size = len(args.gpus.split(','))
    run_script = eval('run_' + mode)

    if world_size > 1:
        port = str(np.random.randint(2048, 65536))
        mp.spawn(run_script, args=(args, port), nprocs=world_size, join=True)
    else:
        run_script(0, args, 0)

def get_snapshot_name(args):
    dirname = '{}_{}'.format(args.method, args.model)
    dirname += '_Ep{}_Bs{}_Lr{}_GPU{}'.format(args.num_epochs, args.batch_size, args.lr, len(args.gpus.split(',')))
    dirname += '_Size{}'.format('x'.join(args.train_size.split(',')))
    return dirname

if __name__ == '__main__':
    image_root = '/home/junsong_fan/diskf/data/cityscape/leftImg8bit_jpg'
    gt_root = '/home/junsong_fan/diskf/data/cityscape/gtFine'

    parser = argparse.ArgumentParser()
    parser.add_argument('--train-image-root', type=str, default=image_root + '/train')
    parser.add_argument('--train-gt-root', type=str, default=gt_root + '/train')
    parser.add_argument('--test-image-root', type=str, default=image_root + '/val')
    parser.add_argument('--test-gt-root', type=str, default=gt_root + '/val')

    parser.add_argument('--model', type=str, default='vgg16_largefov')
    parser.add_argument('--train-size', type=str, default='513,1025')
    parser.add_argument('--test-scales', type=str, default='0.75,1,1.25')
    parser.add_argument('--num-classes', type=int, default=19, help='classification branch num_classes')

    parser.add_argument('--num-epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=16)

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--sgd_mom', type=float, default=0.9)
    parser.add_argument('--sgd_wd', type=float, default=5e-4)

    parser.add_argument('--snapshot', type=str, default='./snapshot/baseline/cityscape')
    parser.add_argument('--num-workers', type=int, default=16)
    parser.add_argument('--gpus', type=str, default='0,1,2,3')
    parser.add_argument('--log-frequency', type=int, default=10)
    parser.add_argument('--pretrained', type=str, default='./pretrained/vgg16_20M.pth')
    parser.add_argument('--force-pretrained', type=str, default='')

    parser.add_argument('--method', type=str, default='point')
    #parser.add_argument('--dilation', type=int, default=1, help="classification branch dilation")
    parser.add_argument('--upsample', type=int, default=1)
    parser.add_argument('--downsample-label', type=int, default=8)

    parser.add_argument('--train-data-list', type=str, default='./scripts/point_labels/Cityscape_1point_0ignore_uniform.txt')

    parser.add_argument('--only-infer', action='store_true')
    args = parser.parse_args()

    #if not args.only_infer:
    #    run(args, 'train')

    run(args, 'extract_feature')
    run_visualize_features(args)

