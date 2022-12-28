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
    dataset = CSPointDataset(args.train_image_root, args.train_data_list, args.train_gt_root, train_size,
            rand_crop=True, rand_mirror=True, rand_scale=(0.5, 0.6), downsample_label=args.downsample_label,
            superpixel_root=args.superpixel_root, max_superpixel=args.max_superpixel)
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

    mod.set_mode('seg_embed')
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
            meter.init('loss_ce', 'loss_ps', 'loss_ss', 'IoU', 'RealIoU')
            confmat = np.zeros((num_classes, num_classes), np.float32)
            real_confmat = np.zeros((num_classes, num_classes), np.float32)
            examples = []
            vid = 0
    
        #for batch, (image, label, real_gt) in enumerate(loader, 1):
        for batch, return_vals in enumerate(loader, 1):
            image, label, real_gt, superpixel = return_vals
            label_cls = F.one_hot(label.to(gpu), 256)[..., :args.num_classes].max(2)[0].max(1)[0].to(torch.float32) # (n, C)
            # forward
            optimizer.zero_grad()

            # cross-entropy loss:
            logit_seg, feature = mod(image.to(gpu))
            logit_ce = logit_seg.mean(dim=(2, 3))
            loss_ce = F.multilabel_soft_margin_loss(logit_ce, label_cls)

            # pooling according to features
            feature = feature / (torch.norm(feature, dim=1, keepdim=True) + 1e-5)
            with torch.no_grad():
                sp_onehot = F.one_hot(superpixel.to(gpu), args.max_superpixel+1)[..., :args.max_superpixel].to(torch.float32)
                ns, hs, ws, _ = sp_onehot.size()

                sp_onehot_flat = sp_onehot.view(ns, hs * ws, args.max_superpixel) # (n, h * w, nS)
                sp_size = sp_onehot_flat.sum(1, keepdim=True)

            nf, df, hf, wf = feature.size()
            feature_flat = feature.view(nf, df, hf * wf) # (n, d, h * w)
            sp_feature = torch.matmul(feature_flat, sp_onehot_flat) / torch.clamp_min(sp_size, 1e-3) # (n, d, nS)
            sp_feature_ = sp_feature.detach()
            
            # pixel-segment CL loss:
            index_pos = sp_onehot_flat
            index_neg = (1 - sp_onehot_flat) * sp_onehot_flat.max(2, keepdim=True)[0] # (n, h * w, nS)

            pixel_segment_dot = torch.matmul(feature_flat.permute(0, 2, 1), sp_feature) # (n, h * w, nS)
            pixel_segment_exp = torch.exp(pixel_segment_dot / args.ps_temp)
            pixel_segment_logit = pixel_segment_exp / (pixel_segment_exp * (index_pos + index_neg + 1e-5)).sum(2, keepdim=True)

            loss_ps = - (index_pos * torch.log(pixel_segment_logit + 1e-5)).sum(2)
            loss_ps = loss_ps.mean()

            # segment prediction loss:
            logit_flat = logit_seg.view(nf, args.num_classes, hf * wf)
            ss_logit = torch.matmul(logit_flat, sp_onehot_flat) / torch.clamp_min(sp_size, 1e-3) # (n, c, nS)
            ss_softmax = torch.softmax(ss_logit, dim=1)
            ss_weight = ss_softmax.max(1, keepdim=True)[0]
            ss_weight = ss_weight / ss_weight.sum(2, keepdim=True)
            ss_merge = (ss_softmax * ss_weight).sum(2)
            loss_ss = F.binary_cross_entropy(ss_merge, label_cls)

            loss = loss_ce * args.lambda_ce + loss_ps * args.lambda_ps + loss_ss * args.lambda_ss
            loss.backward()
            optimizer.step()

            # monitor
            if is_log and (batch % args.log_frequency == 0):
                # loss
                meter.put('loss_ce', loss_ce.item())
                meter.put('loss_ps', loss_ps.item())
                meter.put('loss_ss', loss_ss.item())

                # miou
                with torch.no_grad():
                    # SEG
                    pred = logit_seg.argmax(axis=1).data.cpu().numpy().astype(np.int64)
                    gt = label.data.cpu().numpy().astype(np.int64)
                    gt_index = gt < 255
                    confmat += np.bincount(gt[gt_index] * num_classes + pred[gt_index],
                            minlength=num_classes**2).reshape(num_classes, num_classes)

                    miou = get_iou(confmat).mean()
                    meter.put('IoU', miou)

                    if real_gt.size()[1:] != logit_seg.size()[2:]:
                        real_gt = F.interpolate(torch.unsqueeze(real_gt, 1).float(), logit_seg.size()[2:], mode='nearest').long()
                        real_gt = torch.squeeze(real_gt, 1)
                    real_gt = real_gt.data.cpu().numpy().astype(np.int64)
                    real_gt_index = real_gt < 255
                    real_confmat += np.bincount(real_gt[real_gt_index] * num_classes + pred[real_gt_index],
                            minlength=num_classes**2).reshape(num_classes, num_classes)
                    real_miou = get_iou(real_confmat).mean()
                    meter.put('RealIoU', real_miou)

                    # segment-to-segment
                    sp_exists, sp_lookup = sp_onehot_flat[vid].max(1)
                    sp_ignore = sp_exists < 1e-3
                    sp_label_confidence_ = ss_weight[vid][0][sp_lookup]
                    sp_label_ = ss_logit[vid].argmax(0)[sp_lookup]
                    sp_label_confidence_[sp_ignore] = 0
                    sp_label_[sp_ignore] = 255

                    sp_label_confidence_ = sp_label_confidence_.data.cpu().numpy().reshape(hf, wf)
                    sp_label_ = sp_label_.data.cpu().numpy().reshape(hf, wf)

                # visualization 
                v_img = denormalize_image(image[vid].data.cpu().numpy().transpose(1, 2, 0)[..., ::-1])
                v_real_gt = get_segment_map(real_gt[vid])
                v_gt = get_segment_map(gt[vid])
                v_pred = get_segment_map(pred[vid])

                # 
                v_sp = get_sp_map(superpixel[vid].data.cpu().numpy())
                v_cl_pos = get_score_map((pixel_segment_dot[vid]*index_pos[vid]).sum(1).data.cpu().numpy().reshape(hf, wf), v_img)
                v_cl_neg = get_score_map((pixel_segment_dot[vid]*index_neg[vid]).max(1)[0].data.cpu().numpy().reshape(hf,wf),v_img)

                # segment-to-segment
                v_sp_conf = get_score_map(sp_label_confidence_, v_img)
                v_sp_label = get_segment_map(sp_label_)

                examples_ = [v_img, v_real_gt, v_pred, v_sp, v_cl_pos, v_cl_neg, v_sp_conf, v_sp_label]

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

def run_test(gpu, args, port, verbose=False):
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
    mod = eval(args.model)(args.num_classes, dilation=12, upsample=args.upsample, embedding=args.embedding,
            concat_layers=args.concat_layers.split(',')).to(gpu)
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
    get_pred_src = lambda x: os.path.join(args.snapshot, 'results', 'prediction', os.path.basename(x).split('_')[0],
            '_'.join(os.path.basename(x).split('_')[:3]) + '_pred_labelIds.png')
    for image, label, src in loader:
        print(get_pred_src(src[0]))
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
        pred = pred_logits.argmax(axis=1)
        pred_np = pred[0].data.cpu().numpy().astype(np.uint8)
        pred_np = CS.trainId2id[pred_np.ravel()].reshape(pred_np.shape)

        #city, id1, id2 = src.split('_')[:3]
        #imwrite(os.path.join(args.snapshot, 'results', 'prediction', city,
        #    '{}_{}_{}_pred_labelIds.png'.format(city, id1, id2)), pred_np)
        imwrite(get_pred_src(src[0]), pred_np)

    # demo
    if (gpu == 0):
        get_segment_map = lambda x: CS.palette[x.ravel()].reshape(x.shape+(3,))
        examples = []
        for i, (image, label, src) in enumerate(loader):
            if i >= 20: break
            image = denormalize_image(image.data.cpu().numpy()[0].transpose(1, 2, 0)[..., ::-1])
            gt = get_segment_map(label.numpy()[0])
            examples_ = [image, gt]

            pred_labelIds = cv2.imread(get_pred_src(src[0]), 0)
            pred = CS.paletteId[pred_labelIds.ravel()].reshape(pred_labelIds.shape+(3,))
            examples_.append(pred)

            examples.append(imhstack(examples_, height=240))

        examples = imvstack(examples)
        imwrite(os.path.join(args.snapshot, 'results', 'image_gt_pred.jpg'), examples)

    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()

    # compute mIoU
    if (gpu == 0):
        logger = getLogger(args.snapshot, args.model)
        compute_cs_iou(os.path.join(args.snapshot, 'results', 'prediction'), args.test_gt_root, logger=logger)

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

    dirname += '_{}_{}_sp{}'.format(args.concat_layers.replace(',', '+'), args.embedding, args.max_superpixel)
    dirname += '_T{}'.format(args.ps_temp)
    dirname += '_ce{}_ps{}_ss{}'.format(args.lambda_ce, args.lambda_ps, args.lambda_ss)
    return dirname

if __name__ == '__main__':
    image_root = '/home/junsong_fan/diskf/data/cityscape/leftImg8bit_jpg'
    gt_root = '/home/junsong_fan/diskf/data/cityscape/gtFine'
    sp_root = '/home/junsong_fan/diskf/data/cityscape/leftImg8bit_scg_png'

    parser = argparse.ArgumentParser()
    parser.add_argument('--train-image-root', type=str, default=image_root + '/train')
    parser.add_argument('--train-gt-root', type=str, default=gt_root + '/train')
    parser.add_argument('--test-image-root', type=str, default=image_root + '/val')
    parser.add_argument('--test-gt-root', type=str, default=gt_root + '/val')
    parser.add_argument('--superpixel-root', type=str, default=sp_root + '/train')

    parser.add_argument('--model', type=str, default='vgg16_largefov')
    parser.add_argument('--train-size', type=str, default='513,1025')
    parser.add_argument('--test-scales', type=str, default='0.75,1,1.25')
    parser.add_argument('--num-classes', type=int, default=19, help='classification branch num_classes')

    parser.add_argument('--num-epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=16)

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--sgd_mom', type=float, default=0.9)
    parser.add_argument('--sgd_wd', type=float, default=5e-4)

    #parser.add_argument('--snapshot', type=str, default='./snapshot/segment_to_segment/cityscape')
    parser.add_argument('--snapshot', type=str, default='./snapshot/contrast_superpixel/cityscape')
    parser.add_argument('--num-workers', type=int, default=16)
    parser.add_argument('--gpus', type=str, default='0,1,2,3')
    parser.add_argument('--log-frequency', type=int, default=10)
    parser.add_argument('--pretrained', type=str, default='./pretrained/vgg16_20M.pth')
    parser.add_argument('--force-pretrained', type=str, default='')

    parser.add_argument('--method', type=str, default='point')
    parser.add_argument('--upsample', type=int, default=1)
    parser.add_argument('--downsample-label', type=int, default=8)

    parser.add_argument('--train-data-list', type=str, default='./scripts/point_labels/Cityscape_1point_0ignore_uniform.txt')

    parser.add_argument('--concat-layers', type=str, default='conv5')
    parser.add_argument('--embedding', type=str, default='linear')
    parser.add_argument('--max-superpixel', type=int, default=1024)

    parser.add_argument('--ps-temp', type=float, default=0.1)

    parser.add_argument('--lambda-ce', type=float, default=1)
    parser.add_argument('--lambda-ps', type=float, default=1)
    parser.add_argument('--lambda-ss', type=float, default=1)

    parser.add_argument('--only-infer', action='store_true')
    args = parser.parse_args()

    if not args.only_infer:
        run(args, 'train')

    #run(args, 'test')

