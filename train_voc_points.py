import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from core.data.VOC import VOCPointDataset
from core.model.vgg import vgg16_largefov
from core.model.misc import *
from core.utils import * 
from compute_iou import compute_voc_iou

from core.model.myqueue import *

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

    train_size = [int(x) for x in args.train_size.split(',')]
    dataset = VOCPointDataset(args.train_image_root, args.train_label_file, args.train_gt_root,
            'train_aug', train_size,
            rand_crop=True, rand_mirror=True, rand_scale=(0.5, 1.5),
            downsample_label=args.downsample_label,
            superpixel_root=args.superpixel_root,
            max_superpixel=args.max_superpixel,
            return_image_label=True
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

    mod.set_mode('seg_embed')
    if is_distributed:
        mod = torch.nn.parallel.DistributedDataParallel(mod, device_ids=[gpu], find_unused_parameters=True)

    # others
    if is_log:
        logger = getLogger(args.snapshot, args.model)
        summaryArgs(logger, args, 'green')
        meter = AvgMeter()
        saver = SaveParams(mod, args.snapshot, args.model)
        get_segment_map = lambda x: VOC.palette[x.ravel()].reshape(x.shape+(3,))
        sp_palette = np.random.randint(0, 256, (args.max_superpixel+1, 3), np.uint8)
        get_sp_map = lambda x: sp_palette[x.ravel()].reshape(x.shape+(3,))
        get_iou = lambda x: (np.diag(x) / np.maximum((x.sum(axis=0) + x.sum(axis=1) - np.diag(x)), 1))
        timer = Timer()

    num_batch = len(dataset) // args.batch_size
    lr_scheduler = LrScheduler('poly', args.lr, {'power': 0.9, 'num_epochs': args.num_epochs})
    mod.train()

    # queue
    queue = ClasswiseQueue(args.capacity, args.num_classes)

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
            confmat = np.zeros((num_classes, num_classes), np.float32)
            real_confmat = np.zeros((num_classes, num_classes), np.float32)
            examples = []
            vid = 0
    
        for batch, return_vals in enumerate(loader, 1):
            image, label, real_gt, superpixel, label_class = return_vals
            optimizer.zero_grad()

            # cross-entropy loss
            logit_seg, feature = mod(image.to(gpu))
            loss_ce = F.cross_entropy(logit_seg, label.to(gpu), ignore_index=255)

            # supeprixel aggregate
            n, d, h, w = feature.size()
            feature = feature / (torch.norm(feature, dim=1, keepdim=True) + 1e-5) * args.l2norm_multi
            logit_flat = logit_seg.view(n, args.num_classes, h * w)
            feature_flat = feature.view(n, d, h * w)
            probs_flat = torch.softmax(logit_flat, 1)
            with torch.no_grad():
                #label_onehot = F.one_hot(label.to(gpu), 256)[..., :args.num_classes].to(torch.float32)
                #label_class = label_onehot.max(2)[0].max(1)[0] # (n, c)
                label_class = label_class.to(gpu)
                superpixel = superpixel.view(n, h * w)
                superpixel_onehot = F.one_hot(superpixel.to(gpu), args.max_superpixel+1)\
                        [..., :args.max_superpixel].to(torch.float32)
                superpixel_size = torch.clamp_min(superpixel_onehot.sum(1), 1e-5) # (n, s)
                superpixel_exist = superpixel_onehot.max(2)[0] # (n, h*w)

                # sp seed
                sp_probs = torch.matmul(probs_flat, superpixel_onehot) / superpixel_size.unsqueeze(1) # (n, c, s)
                sp_probs = sp_probs * label_class.unsqueeze(2)
                sp_max_prob, sp_label = sp_probs.max(1)
                sp_label[sp_max_prob < args.sp_label_threshold] = 255 # (n, s)
                sp_label_onehot = F.one_hot(sp_label, 256)[..., :args.num_classes].to(torch.float32) # (n, s, c)
                sp_label_onehot = sp_label_onehot * label_class.unsqueeze(1)
                sp_label_exist = sp_label_onehot.max(2)[0] # (n, s)

                seed_from_sp = (superpixel_onehot * sp_label.unsqueeze(1)).sum(2).to(torch.int64) # (n, h*w)
                seed_from_sp[superpixel_exist < 1e-3] = 255
                seed_from_sp = seed_from_sp.view(n, h, w) # (n, h, w)
            loss_spce = F.cross_entropy(logit_seg, seed_from_sp, ignore_index=255)

            # sp feature, center loss
            with torch.no_grad():
                sp_feat = torch.matmul(feature_flat, superpixel_onehot) / superpixel_size.unsqueeze(1) # (n, d, s)
                class_feat = torch.matmul(
                        sp_feat.permute(1, 0, 2).reshape(d, n * args.max_superpixel), # (d, n*s)
                        sp_label_onehot.view(n * args.max_superpixel, args.num_classes) ) # (n*s, c)
                class_feat_count = sp_label_onehot.sum(dim=(0, 1)) #(c,)
                class_feat = class_feat / torch.clamp_min(class_feat_count, 1e-5).unsqueeze(0) # (d, c)

                if args.memory_method == 'batchwise':
                    class_feat_label = torch.arange(args.num_classes, dtype=torch.int64, device=gpu)
                    class_feat_label[class_feat_count < 1e-3] = 255
                    queue.put(class_feat.T.clone(), class_feat_label, ignore_index=255)

                seed_from_sp_onehot = F.one_hot(seed_from_sp, 256)[..., :args.num_classes].to(torch.float32) # (n, h, w, c)
                feature_center_target = torch.matmul(
                        seed_from_sp_onehot.view(n * h * w, args.num_classes),
                        class_feat.T) # (n * h * w, d)
                feature_center_target = feature_center_target.view(n, h, w, d).permute(0, 3, 1, 2).contiguous() # (n, d, h, w)

                # class_feat exist for the c-th class && seed label exists for th i-th pixel, (n, h, w)
                feature_center_target_exist = (seed_from_sp_onehot * (class_feat_count.view(1, 1, 1, -1) > 1e-3)).max(3)[0]

            if args.spfe_method == 'mse':
                loss_spfe_spatial = ((feature - feature_center_target)**2).sum(1)
            elif args.spfe_method == 'dot':
                loss_spfe_spatial = -(feature * feature_center_target).sum(1)
            else:
                raise RuntimeError(args.spfe_method)
            loss_spfe = (loss_spfe_spatial * feature_center_target_exist).sum() / \
                    torch.clamp_min(feature_center_target_exist.sum(), 1e-5)

            # loss intra-batch cl 
            feature_2d = feature_flat.permute(0, 2, 1).reshape(n * h * w, d)
            feature_dot = torch.matmul(feature_2d, class_feat) # (n*h*w, c)
            loss_incl = F.cross_entropy(feature_dot, seed_from_sp.view(n * h * w), ignore_index=255)

            # loss memory-bank
            feature_mem, feature_mem_label = queue.get() # (q, d), (q,)
            has_mem = feature_mem is not None
            if has_mem:
                q = feature_mem.size(0)
                feature_mem_dot = torch.matmul(feature_2d, feature_mem.T) # (n*h*w, q)
                with torch.no_grad():
                    feature_2d_label_onehot = seed_from_sp_onehot.view(n * h * w, args.num_classes) # (n*h*w, c)
                    #F.one_hot(seed_from_sp.view(n*h*w), 256)[..., :args.num_classes].float() # (n*h*w, c)
                    feature_mem_label_onehot = F.one_hot(feature_mem_label, args.num_classes).float() # (q, c)
                    pos_mask = torch.matmul(feature_2d_label_onehot, feature_mem_label_onehot.T) # (n*h*w, q)
                    neg_mask = (1 - pos_mask) * feature_2d_label_onehot.max(1, keepdim=True)[0]  # (n*h*w, q)

                    if args.memory_rank == 'largest':
                        rank_pos_mask = feature_mem_dot * pos_mask
                        pos_mask = F.one_hot(rank_pos_mask.argmax(1), q) * pos_mask
                    elif args.memory_rank == 'smallest':
                        rank_pos_mask = feature_mem_dot.max(1, keepdim=True)[0] * (1 - pos_mask) + \
                                feature_mem_dot * pos_mask
                        pos_mask = F.one_hot(rank_pos_mask.argmin(1), q) * pos_mask
                    elif args.memory_rank == 'none':
                        pass
                    else:
                        raise RuntimeError(args.memory_rank)

                feature_mem_dot_exp = torch.exp(feature_mem_dot)
                logit_mem = feature_mem_dot_exp / torch.clamp_min(
                        (feature_mem_dot_exp * (pos_mask + neg_mask)).sum(1, keepdim=True), 1e-5)
                loss_mem = - (torch.log(logit_mem + 1e-5) * pos_mask).sum() / torch.clamp_min(pos_mask.sum(), 1e-5)
            else:
                loss_mem = 0

            # backward
            upper_iter = num_batch * 3.
            rampup = min((num_batch * epoch + batch) / upper_iter, 1.0)
            rampup = np.exp((rampup - 1) * 5)
            lambda_spce = args.lambda_spce * rampup
            lambda_spfe = args.lambda_spfe * rampup
            lambda_incl = args.lambda_incl * rampup
            lambda_mem = args.lambda_mem * rampup

            loss = loss_ce * args.lambda_ce + loss_spce * lambda_spce + \
                    loss_spfe * lambda_spfe + loss_incl * lambda_incl + \
                    loss_mem * lambda_mem

            #print(gpu, 'ce,spce,spfe,incl,mem:', loss_ce.item(), loss_spce.item(), loss_spfe.item(), loss_incl.item(), loss_mem.item() if isinstance(loss_mem, torch.Tensor) else loss_mem)

            loss.backward()
            optimizer.step()

            # monitor
            if is_log and (batch % args.log_frequency == 0):
                # loss
                meter.put('loss_ce', loss_ce.item())
                meter.put('loss_spce', loss_spce.item())
                meter.put('loss_spfe', loss_spfe.item())
                meter.put('loss_incl', loss_incl.item())
                meter.put('loss_mem', loss_mem.item())
                if has_mem:
                    meter.put('num_pos', pos_mask.sum().item()/(n * h * w))
                    meter.put('num_neg', neg_mask.sum().item()/(n * h * w))

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

                    feature_dot_item = (feature_dot.view(n, h, w, args.num_classes)[vid] * seed_from_sp_onehot[vid])
                    feature_dot_item = feature_dot_item.sum(2).data.cpu().numpy() # (h, w)

                    if has_mem:
                        feature_mem_dot_item = feature_mem_dot.view(n, h, w, -1)[vid] * pos_mask.view(n,h,w,-1)[vid]
                        feature_mem_dot_item = feature_mem_dot_item.sum(2).data.cpu().numpy() # (h, w)

                # visualization 
                v_img = denormalize_image(image[vid].data.cpu().numpy().transpose(1, 2, 0)[..., ::-1])
                v_real_gt = get_segment_map(real_gt[vid])
                v_gt = get_segment_map(gt[vid])
                v_pred = get_segment_map(pred[vid])
                v_seed_from_sp = get_segment_map(seed_from_sp[vid].data.cpu().numpy())

                #
                v_spfe_spatial = get_score_map(loss_spfe_spatial[vid].data.cpu().numpy(), v_img)
                v_spfe_exists = get_score_map(feature_center_target_exist[vid].data.cpu().numpy(), v_img)
                v_feature_dot = get_score_map(feature_dot_item, v_img)

                examples_ = [v_img, v_real_gt, v_gt, v_pred] + \
                        [v_seed_from_sp] + \
                        [v_spfe_spatial, v_spfe_exists, v_feature_dot]

                if has_mem:
                    v_feature_mem_dot = get_score_map(feature_mem_dot_item, v_img)
                    examples_.append(v_feature_mem_dot)

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
    dataset = VOCPointDataset(args.test_image_root, None, args.test_gt_root, 'val',
            target_size=None, return_src=True)
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
    get_pred_src = lambda x: os.path.join(args.snapshot, 'results', 'prediction',
            os.path.basename(x).split('.')[0] + '.png')

    for image, label, src in loader:
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

        imwrite(get_pred_src(src[0]), pred_np)

    # demo
    if (gpu == 0):
        get_segment_map = lambda x: VOC.palette[x.ravel()].reshape(x.shape+(3,))
        examples = []
        for i, (image, label, src) in enumerate(loader):
            if i >= 20: break
            image = denormalize_image(image.data.cpu().numpy()[0].transpose(1, 2, 0)[..., ::-1])
            gt = get_segment_map(label.numpy()[0])
            pred = get_segment_map(cv2.imread(get_pred_src(src[0]), 0))

            examples_ = [image, gt, pred]
            examples.append(imhstack(examples_, height=240))

        examples = imvstack(examples)
        imwrite(os.path.join(args.snapshot, 'results', 'image_gt_pred.jpg'), examples)

    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()

    # compute mIoU
    if (gpu == 0):
        logger = getLogger(args.snapshot, args.model)
        compute_voc_iou(
                os.path.join(args.snapshot, 'results', 'prediction'),
                args.test_gt_root, 'val', logger=logger)

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
    if args.embedding != 'linear':
        dirname += '_' + args.embedding
    if args.concat_layers != 'conv5,fc6,fc7':
        dirname += '_' + args.concat_layers.replace(',', '+')

    dirname += '_l2m{}'.format(args.l2norm_multi)
    if args.lambda_spfe > 0 and args.spfe_method != 'mse':
        dirname += '_' + args.spfe_method
    if args.lambda_mem > 0:
        dirname += '_MEM{}_{}'.format(args.capacity, args.memory_method)
        if args.memory_rank != 'none':
            dirname += '_' + args.memory_rank

    dirname += '_ce{}_spth{}_spce{}_spfe{}_incl{}_mem{}'.format(args.lambda_ce, args.sp_label_threshold, args.lambda_spce, args.lambda_spfe, args.lambda_incl, args.lambda_mem)
    if args.suffix:
        dirname += '_' + args.suffix
    return dirname

if __name__ == '__main__':
    image_root = '/home/junsong_fan/diskf/data/cityscape/leftImg8bit_jpg'
    gt_root = '/home/junsong_fan/diskf/data/cityscape/gtFine'
    sp_root = '/home/junsong_fan/diskf/data/cityscape/leftImg8bit_scg_png'

    parser = argparse.ArgumentParser()
    parser.add_argument('--train-image-root', type=str, default=\
            '/home/junsong_fan/diskf/data/VOC2012/JPEGImages')
    parser.add_argument('--train-gt-root', type=str, default=\
            '/home/junsong_fan/diskf/data/VOC2012/extra/SegmentationClassAug')
    parser.add_argument('--test-image-root', type=str, default=\
            '/home/junsong_fan/diskf/data/VOC2012/JPEGImages')
    parser.add_argument('--test-gt-root', type=str, default=\
            '/home/junsong_fan/diskf/data/VOC2012/extra/SegmentationClassAug')
    parser.add_argument('--train-label-file', type=str, default=\
            './resources/whats_the_point/train_aug_points_gtBackground.txt')
    parser.add_argument('--superpixel-root', type=str, default=\
            '/home/junsong_fan/diskf/data/VOC2012/extra/Oversegment')

    parser.add_argument('--model', type=str, default='vgg16_largefov')
    parser.add_argument('--train-size', type=str, default='321,321')
    parser.add_argument('--test-scales', type=str, default='0.75,1,1.25')
    parser.add_argument('--num-classes', type=int, default=21, help='classification branch num_classes')

    parser.add_argument('--num-epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=16)

    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--sgd_mom', type=float, default=0.9)
    parser.add_argument('--sgd_wd', type=float, default=5e-4)

    parser.add_argument('--snapshot', type=str, default='./snapshot/pre_ablation/voc')
    parser.add_argument('--num-workers', type=int, default=16)
    parser.add_argument('--gpus', type=str, default='0,1,2,3')
    parser.add_argument('--log-frequency', type=int, default=35)
    parser.add_argument('--pretrained', type=str, default='./pretrained/vgg16_20M.pth')
    parser.add_argument('--force-pretrained', type=str, default='')

    parser.add_argument('--method', type=str, default='point')
    parser.add_argument('--upsample', type=int, default=1)
    parser.add_argument('--downsample-label', type=int, default=8)

    parser.add_argument('--concat-layers', type=str, default='fc7')
    parser.add_argument('--embedding', type=str, default='mlp')
    parser.add_argument('--max-superpixel', type=int, default=1024)

    parser.add_argument('--suffix', type=str, default='')

    parser.add_argument('--sp-label-threshold', type=float, default=0.1)

    parser.add_argument('--spfe-method', type=str, default='dot')
    parser.add_argument('--l2norm-multi', type=float, default=3.8)
    
    parser.add_argument('--capacity', type=int, default=64)
    parser.add_argument('--memory-method', type=str, default='batchwise')
    parser.add_argument('--memory-rank', type=str, default='none')

    parser.add_argument('--lambda-ce', type=float, default=1)
    parser.add_argument('--lambda-spce', type=float, default=1)
    parser.add_argument('--lambda-spfe', type=float, default=0)
    parser.add_argument('--lambda-incl', type=float, default=0)
    parser.add_argument('--lambda-mem', type=float, default=1)

    parser.add_argument('--only-infer', action='store_true')
    args = parser.parse_args()

    if not args.only_infer:
        run(args, 'train')

    run(args, 'test')

