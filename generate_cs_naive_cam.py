import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

import multiprocessing

from core.data.Cityscape import CSPointDataset, CSClassDataset
from core.model.vgg import vgg16_largefov
from core.model.misc import *
from core.utils import * 
from compute_iou import compute_cs_iou

import argparse
import os
import copy


def run_train(gpu, args, port):
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
    train_size = [int(x) for x in args.train_size.split(',')]
    dataset = CSClassDataset(args.train_image_root, args.train_label_list,
            train_size,
            rand_crop=True, rand_mirror=True, rand_scale=(0.75, 1.25),
            segment_root=args.train_gt_root
    )
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True) if is_distributed else None
    device_bs = args.batch_size // world_size
    loader = DataLoader(dataset, batch_size=device_bs, shuffle=sampler is None, pin_memory=False,
            drop_last=True, sampler=sampler, num_workers=(args.num_workers+world_size-1) // world_size)

    # model
    mod = eval(args.model)(args.num_classes, dilation=12, upsample=args.upsample,
            embedding=args.embedding, concat_layers=args.concat_layers.split(',')).to(gpu)
    pretrained = torch.load(args.pretrained, map_location={'cuda:0': 'cuda:%s'%gpu})
    mod.init_params(pretrained, args.seed)

    lr_mult_params = mod.get_param_groups()
    param_groups = []
    for lr_mult, params in lr_mult_params.items():
        param_groups.append({'params': params, 'lr': args.lr * lr_mult})
    optimizer = torch.optim.SGD(param_groups, lr=args.lr, momentum=args.sgd_mom, weight_decay=args.sgd_wd)

    mod.set_mode('seg')
    if is_distributed:
        mod = torch.nn.parallel.DistributedDataParallel(mod, device_ids=[gpu], find_unused_parameters=True)

    # others
    if is_log:
        logger = getLogger(args.snapshot, args.model)
        summaryArgs(logger, args, 'green')
        meter = AvgMeter()
        saver = SaveParams(mod, args.snapshot, args.model)
        get_segment_map = lambda x: CS.palette[x.ravel()].reshape(x.shape+(3,))
        sp_palette = np.random.randint(0, 256, (args.max_superpixel+1, 3), np.uint8)
        get_sp_map = lambda x: sp_palette[x.ravel()].reshape(x.shape+(3,))
        get_iou = lambda x: (np.diag(x) / np.maximum((x.sum(axis=0) + x.sum(axis=1) - np.diag(x)), 1))
        timer = Timer()

    lr_scheduler = LrScheduler('poly', args.lr, {'power': 0.9, 'num_epochs': args.num_epochs})
    mod.train()

    # train
    for epoch in range(args.num_epochs):
        # reset loader
        if is_distributed:
            sampler.set_epoch(epoch)

        # set lr
        lr = lr_scheduler.get(epoch)
        for lr_mult, param_group in zip(lr_mult_params.keys(), optimizer.param_groups):
            param_group['lr'] = lr * lr_mult
            if is_log:
                info(logger, 'Set lr={} for {} params.'.format(lr * lr_mult, len(param_group['params'])), 'yellow')

        if is_log:
            num_classes = args.num_classes
            meter.init('loss_ce', 'IoU')
            confmat = np.zeros((num_classes, num_classes), np.float32)
            real_confmat = np.zeros((num_classes, num_classes), np.float32)
            examples = []
            vid = 0
    
        for batch, return_vals in enumerate(loader, 1):
            image, label, gt = return_vals
            optimizer.zero_grad()

            # forward
            logit = mod(image.to(gpu))
            logit_gap = logit.mean(dim=(2, 3))
            loss_ce = F.multilabel_soft_margin_loss(logit_gap, label.to(gpu))

            loss = loss_ce

            # backward
            loss.backward()
            optimizer.step()

            # monitor
            if is_log and (batch % args.log_frequency == 0):
                # loss
                meter.put('loss_ce', loss_ce.item())

                # miou
                with torch.no_grad():
                    # prediction
                    pred = logit.argmax(axis=1).data.cpu().numpy().astype(np.int64)
                    if gt.size()[1:] != logit.size()[2:]:
                        gt = F.interpolate(gt.unsqueeze(1).float(), logit.size()[2:], mode='nearest').long()
                        gt = torch.squeeze(gt, 1)

                    gt = gt.data.cpu().numpy().astype(np.int64)
                    gt_index = gt < 255
                    confmat += np.bincount(gt[gt_index] * num_classes + pred[gt_index],
                            minlength=num_classes**2).reshape(num_classes, num_classes)

                    miou = get_iou(confmat).mean()
                    meter.put('IoU', miou)

                    # predicted CAM score
                    pred_score = torch.clamp_min(logit[vid], 0)
                    pred_score = pred_score / torch.clamp_min(pred_score.max(2,keepdim=True)[0].max(1,keepdim=True)[0], 1e-5)
                    pred_score = pred_score.cpu() * label[vid].view(num_classes, 1, 1)
                    pred_score = pred_score.data.cpu().numpy()


                #
                v_img = denormalize_image(image[vid].data.cpu().numpy().transpose(1, 2, 0)[..., ::-1])
                v_gt = get_segment_map(gt[vid])
                v_pred = get_segment_map(pred[vid])
                v_pred_score = get_score_map(pred_score.max(axis=0), v_img)

                examples_ = [v_img, v_gt, v_pred, v_pred_score]
                examples.append(imhstack(examples_, height=240))
                imwrite(os.path.join(args.snapshot, 'train_demo_preview.jpg'), examples[-1])
                vid = (vid + 1) % device_bs

                timer.record()
                info(logger, 'Epoch={}, Batch={}, {}, Speed={:.1f} img/sec'.format(
                    epoch, batch, meter, args.log_frequency*args.batch_size/timer.interval()) )

        # log & save
        if is_log:
            examples = imvstack(examples)
            imwrite(os.path.join(args.snapshot, 'train_demo', '%s-%04d.jpg' % (args.model, epoch)), examples)
            saver(epoch)
            info(logger, 'Saved params to: {}'.format(saver.filename), 'yellow')

            iou = get_iou(confmat)
            info(logger, 'Epoch={}, IoU={}\n{}'.format(epoch, iou.mean(), iou) )
            real_iou = get_iou(real_confmat)
            info(logger, 'Epoch={}, RealIoU={}\n{}'.format(epoch, real_iou.mean(), real_iou) )

    if is_distributed:
        dist.destroy_process_group()

def run_cam(gpu, args, port, verbose=False):
    # env
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    world_size = len(args.gpus.split(','))
    is_distributed = world_size > 1

    if is_distributed:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = port
        dist.init_process_group('nccl', rank=gpu, world_size=world_size)
    is_log = gpu == 0

    #data
    info(None, args.test_image_root, 'red')
    info(None, args.test_label_list, 'red')
    dataset = CSClassDataset(args.test_image_root, args.test_label_list, None,
            return_src=True, use_city_name=True)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False) if is_distributed else None
    loader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=False, drop_last=False,
            sampler=sampler, num_workers=1)

    # model
    mod = eval(args.model)(args.num_classes, dilation=12, upsample=args.upsample,
            embedding=args.embedding, concat_layers=args.concat_layers.split(',')).to(gpu)
    mod.set_mode('seg')
    if args.force_pretrained:
        pretrained = args.force_pretrained
    else:
        pretrained = os.path.join(args.snapshot, 'checkpoint', '%s-%04d.pth' % (args.model, args.num_epochs-1))
    pretrained = torch.load(pretrained, map_location={'cuda:0': 'cuda:%s'%gpu})
    if is_distributed:
        mod = torch.nn.parallel.DistributedDataParallel(mod, device_ids=[gpu], find_unused_parameters=True)
    mod.load_state_dict(pretrained, strict=True)
    mod.train(False)

    # ms test
    test_scales = [float(x) for x in args.test_scales.split(',')]
    get_save_src = lambda x: os.path.join(args.snapshot, 'results', 'cam',
            '_'.join(os.path.basename(x).split('_')[:3]) + '.npy')
    #get_save_src = lambda x: os.path.join(args.snapshot, 'results', 'threshold_seed_{}'.format(args.cam_threshold),
    #        os.path.basename(x).split('_')[0],
    #        '_'.join(os.path.basename(x).split('_')[:3]) + '_thSeed_labelIds.png')
    for image, label, src in loader:
        print(src[0])
        image = image.to(gpu)
        _, _, h, w = image.size()
        pred_logits = []

        with torch.no_grad():
            for scale in test_scales:
                image_ = F.interpolate(image, scale_factor=scale, mode='bilinear', align_corners=True) if scale != 1 else image
                image_flip = torch.flip(image_, (3,))
                image_in = torch.cat([image_, image_flip], dim=0)
                pred = mod(image_in)
                pred = F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=True)
                pred = (pred[0:1] + torch.flip(pred[1:2], (3,))) / 2
                pred_logits.append(pred)

        pred_logits = sum(pred_logits) / len(pred_logits)
        Ls = np.nonzero(label.data.cpu().numpy().ravel())[0]
        pred_logits = pred_logits[0][Ls].data.cpu().numpy()
        npsave(get_save_src(src[0]), pred_logits)

        #cam = np.maximum(pred_logits, 0)

    # demo
    if (gpu == 0):
        get_segment_map_from_id = lambda x: CS.paletteId[x.ravel()].reshape(x.shape+(3,))
        get_gt_src = lambda x: os.path.join(args.test_gt_root, os.path.basename(x).split('_')[0],
                '_'.join(os.path.basename(x).split('_')[:3]) + '_gtFine_labelIds.png')
        load_gt = lambda x: get_segment_map_from_id(cv2.imread(get_gt_src(x), 0))

        examples = []
        for i, (image, label, src) in enumerate(loader):
            if i >= 20: break
            print(get_save_src(src[0]))
            print(get_gt_src(src[0]))
            image = denormalize_image(image.data.cpu().numpy()[0].transpose(1, 2, 0)[..., ::-1])
            gt = load_gt(src[0])
            Ls = np.nonzero(label.data.cpu().numpy().ravel())[0]
            cam = np.load(get_save_src(src[0]))
            assert len(Ls) == len(cam), (src, Ls, cam.shape)

            examples_ = [image, gt]

            if len(Ls) > 0:
                cam_list = []
                cam = np.maximum(cam, 0)
                cam /= np.maximum(cam.max(axis=(1, 2), keepdims=True), 1e-5)
                for i, L in enumerate(Ls):
                    this_cam = get_score_map(cam[i], image)
                    this_cam = imtext(this_cam, 'class_%d'%L)
                    cam_list.append(this_cam)

                cam_list = patch_images(cam_list, 2, 5, cam_list[0].shape)
                examples_.append(cam_list)

            examples.append(imhstack(examples_, height=240))

        examples = imvstack(examples)
        imwrite(os.path.join(args.snapshot, 'results', 'image_gt_cam.jpg'), examples)

    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()

def _generate_threshold_seed_thread(file_root, save_root, name, Ls):
    city, id0, id1 = name.split('_')[:3]
    cam_file = os.path.join(file_root, '{}_{}_{}.npy'.format(city, id0, id1))
    assert os.path.exists(cam_file), cam_file
    cam = np.load(cam_file)
    assert cam.shape == (len(Ls), 1024, 2048), (name, cam.shape, Ls)

    cam = np.maximum(cam, 0)
    cam = cam / np.maximum(cam.max(axis=(1, 2), keepdims=True), 1e-5)
    cam_label = cam.argmax(axis=0)
    cam_label = Ls[cam_label.ravel()].reshape(cam_label.shape)

    cam_score = cam.max(axis=0)
    unsure_region = cam_score < args.cam_threshold
    cam_label[unsure_region] = 255
    cam_label = CS.trainId2id[cam_label.ravel()].reshape(cam_label.shape)
    imwrite(os.path.join(save_root, city, '{}_{}_{}_thSeed_labelIds.png'.format(city, id0, id1)), cam_label)

def _tmp_convert_trainId_to_id(file_root, save_root, name, Ls):
    city, id0, id1 = name.split('_')[:3]
    src = os.path.join(save_root, city, '{}_{}_{}_thSeed_labelIds.png'.format(city, id0, id1))
    assert os.path.exists(src), src

    seed = cv2.imread(src)
    seed2 = CS.trainId2id[seed.ravel()].reshape(seed.shape)
    imwrite(src, seed2)

def generate_threshold_seed(args):
    args.snapshot = os.path.join(args.snapshot, get_snapshot_name(args))
    file_root = os.path.join(args.snapshot, 'results', 'cam')
    save_root = os.path.join(args.snapshot, 'results', 'threshold_seed_{}'.format(args.cam_threshold))
    with open(args.test_label_list) as f:
        name_labels = [x.strip().split(' ') for x in f.readlines()]
    names = [x[0] for x in name_labels]
    labels = [np.array([int(L) for L in x[1:]]) for x in name_labels]
    pool = multiprocessing.Pool(32)
    jobs = []
    for name, Ls in zip(names, labels):
        jobs.append(pool.apply_async(_generate_threshold_seed_thread, (file_root, save_root, name, Ls)))
        #jobs.append(pool.apply_async(_tmp_convert_trainId_to_id, (file_root, save_root, name, Ls)))
    [job.get() for job in jobs]

    # demo
    examples = []
    get_segment_map_from_id = lambda x: CS.paletteId[x.ravel()].reshape(x.shape+(3,))
    get_segment_map = lambda x: CS.palette[x.ravel()].reshape(x.shape+(3,))
    for i, (name, Ls) in enumerate(zip(names, labels)):
        if i >= 20: break
        city, id0, id1 = name.split('_')[:3]
        img_src = os.path.join(args.test_image_root, city, '{}_{}_{}_leftImg8bit.jpg'.format(city, id0, id1))
        gt_src = os.path.join(args.test_gt_root, city, '{}_{}_{}_gtFine_labelIds.png'.format(city, id0, id1))
        seed_src = os.path.join(save_root, city, '{}_{}_{}_thSeed_labelIds.png'.format(city, id0, id1))
        assert os.path.exists(img_src), img_src
        assert os.path.exists(gt_src), gt_src
        assert os.path.exists(seed_src), seed_src

        img = cv2.imread(img_src)
        gt = get_segment_map_from_id(cv2.imread(gt_src, 0))
        seed = get_segment_map_from_id(cv2.imread(seed_src, 0))
        examples.append(imhstack([img, gt, seed], height=360))
    examples = imvstack(examples)
    imwrite(save_root + '_demo.jpg', examples)

def run(args, mode):
    args = copy.copy(args)
    args.snapshot = os.path.join(args.snapshot, get_snapshot_name(args))
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

    #dirname += '_{}_{}_sp{}'.format(args.concat_layers.replace(',', '+'), args.embedding, args.max_superpixel)
    #dirname += '_T{}'.format(args.ps_temp)
    #dirname += '_ce{}_ps{}_ss{}'.format(args.lambda_ce, args.lambda_ps, args.lambda_ss)
    return dirname

if __name__ == '__main__':
    image_root = '/home/junsong_fan/diskf/data/cityscape/leftImg8bit_jpg'
    gt_root = '/home/junsong_fan/diskf/data/cityscape/gtFine'
    sp_root = '/home/junsong_fan/diskf/data/cityscape/leftImg8bit_scg_png'

    parser = argparse.ArgumentParser()
    parser.add_argument('--train-image-root', type=str, default=\
            '/home/junsong_fan/diskf/data/cityscape/patches/leftImg8bit_jpg/train')
    parser.add_argument('--train-label-list', type=str, default=\
            './scripts/image_labels/gtFine_patch_400x500_n15_imageLabels.txt')
    parser.add_argument('--train-gt-root', type=str, default=\
            '/home/junsong_fan/diskf/data/cityscape/patches/gtFine/train')

    parser.add_argument('--test-image-root', type=str, default=image_root + '/train')
    parser.add_argument('--test-gt-root', type=str, default=gt_root + '/train')
    parser.add_argument('--test-label-list', type=str, default=\
            './scripts/image_labels/gtFine_1024x2048_imageLabels.txt')
    parser.add_argument('--superpixel-root', type=str, default=sp_root + '/train')

    parser.add_argument('--model', type=str, default='vgg16_largefov')
    #parser.add_argument('--train-size', type=str, default='513,1025')
    parser.add_argument('--train-size', type=str, default='400,500')
    parser.add_argument('--test-scales', type=str, default='0.75,1,1.25')
    parser.add_argument('--num-classes', type=int, default=19, help='classification branch num_classes')

    parser.add_argument('--num-epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=16)

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--sgd_mom', type=float, default=0.9)
    parser.add_argument('--sgd_wd', type=float, default=5e-4)

    #parser.add_argument('--snapshot', type=str, default='./snapshot/segment_to_segment/cityscape')
    parser.add_argument('--snapshot', type=str, default='./snapshot/naive_cam/cityscape')
    parser.add_argument('--num-workers', type=int, default=16)
    parser.add_argument('--gpus', type=str, default='0,1,2,3')
    parser.add_argument('--log-frequency', type=int, default=100)
    parser.add_argument('--pretrained', type=str, default='./pretrained/vgg16_20M.pth')
    parser.add_argument('--force-pretrained', type=str, default='')

    parser.add_argument('--method', type=str, default='point')
    parser.add_argument('--upsample', type=int, default=1)
    parser.add_argument('--downsample-label', type=int, default=8)

    #parser.add_argument('--train-data-list', type=str, default='./scripts/point_labels/Cityscape_1point_0ignore_uniform.txt')

    parser.add_argument('--concat-layers', type=str, default='conv5')
    parser.add_argument('--embedding', type=str, default='linear')
    parser.add_argument('--max-superpixel', type=int, default=1024)
    parser.add_argument('--cam-threshold', type=float, default=0.3)

    #parser.add_argument('--ps-temp', type=float, default=0.1)

    #parser.add_argument('--lambda-ce', type=float, default=1)
    #parser.add_argument('--lambda-ps', type=float, default=1)
    #parser.add_argument('--lambda-ss', type=float, default=1)

    parser.add_argument('--only-infer', action='store_true')
    args = parser.parse_args()

    #if not args.only_infer:
    #    run(args, 'train')

    #run(args, 'cam')

    generate_threshold_seed(args) 

