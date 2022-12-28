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

from core.model.projection_model import *

_SP_PALETTE = np.random.randint(0, 256, (2048, 3), np.uint8)
_USE_PROJ_MODEL = True

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
    if _USE_PROJ_MODEL:
        mod = ProjectionModel(base="DeeplabV2_VGG16", base_args={
            "num_classes": 21,
            "pretrained": "./pretrained/vgg16_20M.pth"
        },
        base_stage_names=["c7"],
        base_stage_dims=[1024],
        proj_mid_dims=512,
        proj_out_dims=256,
        normalize=False
        ).to(gpu)
        params_lr_mult = get_params_lr_mult(mod, [['fc8', 10]])
        param_groups = [{'params': params, 'lr': args.lr*mult}
            for mult, params in params_lr_mult.items()]
        optimizer = torch.optim.SGD(param_groups, lr=args.lr, momentum=args.sgd_mom, weight_decay=args.sgd_wd)
        lr_mult_params = {mult: None for mult in params_lr_mult.keys()}
    else:
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
    num_queues = len(args.mem_type.split(',')) if args.separate else 1
    capacities = [int(capacity) for capacity in args.capacity.split(',')]
    if len(capacities) == 1:
        capacities = [capacities[0]] * num_queues
    assert len(capacities) == num_queues, (capacities, num_queues)

    queues = [ClasswiseQueue(capacity, args.num_classes) for capacity in capacities]

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
            superpixel_view = superpixel
            optimizer.zero_grad()

            # cross-entropy loss
            logit_seg, feature = mod(image.to(gpu))
            if True:
                loss_ce = F.cross_entropy(logit_seg, label.to(gpu), ignore_index=255, reduction='sum') / \
                        torch.clamp_min((label.to(gpu) < args.num_classes).float().sum(), 1e-5)
            else:
                loss_ce = F.cross_entropy(logit_seg, label.to(gpu), ignore_index=255)

            # supeprixel aggregate
            n, d, h, w = feature.size()
            feature = feature / (torch.norm(feature, dim=1, keepdim=True) + 1e-5)
            logit_flat = logit_seg.view(n, args.num_classes, h * w)
            feature_flat = feature.view(n, d, h * w)
            probs_flat = torch.softmax(logit_flat, 1)
            with torch.no_grad():
                label_class = label_class.to(gpu)
                superpixel = superpixel.view(n, h * w)
                superpixel_onehot = F.one_hot(superpixel.to(gpu), args.max_superpixel+1)\
                        [..., :args.max_superpixel].to(torch.float32) # (n, h*w, s)
                superpixel_size = torch.clamp_min(superpixel_onehot.sum(1), 1e-5) # (n, s)
                superpixel_exist = superpixel_onehot.max(2)[0] # (n, h*w)

                # sp seed
                sp_probs = torch.matmul(probs_flat, superpixel_onehot) / superpixel_size.unsqueeze(1) # (n, c, s)
                sp_probs = sp_probs * label_class.unsqueeze(2)
                sp_max_prob, sp_label = sp_probs.max(1)
                sp_label[sp_max_prob < args.sp_label_threshold] = 255 # (n, s)
                sp_label_onehot = F.one_hot(sp_label, 256)[..., :args.num_classes].to(torch.float32) # (n, s, c)
                sp_label_onehot = sp_label_onehot * label_class.unsqueeze(1)
                sp_label_exist, sp_label = sp_label_onehot.max(2) # (n, s)
                sp_label[sp_label_exist < 1e-3] = 255

                seed_from_sp = (superpixel_onehot * sp_label.unsqueeze(1)).sum(2).to(torch.int64) # (n, h*w)
                seed_from_sp[superpixel_exist < 1e-3] = 255
                seed_from_sp = seed_from_sp.view(n, h, w) # (n, h, w)
            loss_spce = F.cross_entropy(logit_seg, seed_from_sp, ignore_index=255)

            with torch.no_grad():
                # superpixel-level feature
                pix_feat = feature_flat.permute(0, 2, 1).reshape(n * h * w, d)
                pix_feat_flat = pix_feat
                #pix_feat_prob, pix_feat_label = probs_flat.max(1)
                pix_feat_prob, pix_feat_label = (probs_flat * label_class.unsqueeze(2)).max(1)
                pix_feat_label[pix_feat_prob < 0.9] = 255
                pix_feat_label = pix_feat_label.view(-1)

                sp_feat = torch.matmul(feature_flat, superpixel_onehot) / superpixel_size.unsqueeze(1) # (n, d, s)
                sp_feat_flat = sp_feat.permute(0, 2, 1).reshape(n * args.max_superpixel, d) # (n*s, d)
                #sp_feat_label = sp_label.view(n * args.max_superpixel) # (n*s,)
                sp_feat_prob, sp_feat_label = sp_probs.max(1)
                sp_feat_label[sp_feat_prob < 0.9] = 255
                sp_feat_label = sp_feat_label.view(n * args.max_superpixel)

                # image-level feature
                some_label = pix_feat_label
                if args.use_intersection:
                    some_label[seed_from_sp.view(-1) != pix_feat_label] = 255
                some_feat_label_onehot = F.one_hot(some_label.view(n, -1), 256)[..., :args.num_classes].float() # (n, h*w, c)
                img_item_cnt = torch.clamp_min(some_feat_label_onehot.sum(1), 1e-5) # (n, c)
                img_feat = torch.matmul(feature_flat, some_feat_label_onehot) / img_item_cnt.unsqueeze(1) # (n, d, c)
                img_feat_flat = img_feat.permute(0, 2, 1).reshape(n * args.num_classes, d) # (n*c, d)
                img_feat_label = torch.arange(args.num_classes, dtype=torch.int64, device=gpu).repeat(n, 1) # (n, c)
                img_feat_label[img_item_cnt < 5] = 255
                img_feat_label = img_feat_label.view(n * args.num_classes)

                # superpixel-image-level feature
                #spimg_item_cnt = torch.clamp_min(sp_label_onehot.sum(1), 1e-5) # (n, c)
                #spimg_feat = torch.matmul(sp_feat, sp_label_onehot) / spimg_item_cnt.unsqueeze(1) # (n, d, c)
                #spimg_feat_flat = spimg_feat.permute(0, 2, 1).reshape(n * args.num_classes, d) # (n*c, d)
                #spimg_feat_label = torch.arange(args.num_classes, dtype=torch.int64, device=gpu).repeat(n, 1)
                #spimg_feat_label[spimg_item_cnt < 1e-3] = 255 # (n, c)
                #spimg_feat_label = spimg_feat_label.view(n * args.num_classes) # (n*c,)

                # superpixel-batch-level feature
                #spbth_item_cnt = torch.clamp_min(sp_label_onehot.sum(dim=(0, 1)), 1e-5) # (c,)
                #spbth_feat = torch.matmul(sp_label_onehot.view(n * args.max_superpixel, args.num_classes).T,
                #        sp_feat_flat) / spbth_item_cnt.unsqueeze(1) # (c, d)
                #spbth_feat_flat = spbth_feat
                #spbth_feat_label = torch.arange(args.num_classes, dtype=torch.int64, device=gpu)
                #spbth_feat_label[spbth_item_cnt < 1e-3] = 255 # (c,)

                # superpixel-image-batch-level feature
                #spimg_feat_label_onehot = F.one_hot(spimg_feat_label, 256)[..., :args.num_classes].to(torch.float32) # (n*c, c)
                #spimgbth_item_cnt = torch.clamp_min(spimg_feat_label_onehot.sum(0), 1e-5) # (c,)
                #spimgbth_feat = torch.matmul(spimg_feat_label_onehot.T, spimg_feat_flat) / spimgbth_item_cnt.unsqueeze(1) # (c, d)
                #spimgbth_feat_flat = spimgbth_feat
                #spimgbth_feat_label = torch.arange(args.num_classes, dtype=torch.int64, device=gpu)
                #spimgbth_feat_label[spimgbth_item_cnt < 1e-3] = 255 # (c,)

                # Enqueue
                enq_feats, enq_labels = [], []
                for i, mem_type in enumerate(args.mem_type.split(',')):
                    enq_feat = eval(mem_type + '_feat_flat')
                    enq_label = eval(mem_type + '_feat_label')
                    if args.separate:
                        queues[i].put(enq_feat, enq_label)
                    else:
                        enq_feats.append(enq_feat)
                        enq_labels.append(enq_label)
                if not args.separate:
                    enq_feat = torch.cat(enq_feats, 0)
                    enq_label = torch.cat(enq_labels, 0)
                    queues[0].put(enq_feat, enq_label)

            # Dequeue
            #deq_feat, deq_label = queue.get() # (q, d), (q,)
            #has_mem = deq_feat is not None
            deq_results = [queue.get() for queue in queues]
            has_mem = all([deq_result[0] is not None for deq_result in deq_results])

            # loss cl
            if has_mem:
                loss_mems = []
                for deq_feat, deq_label in deq_results:
                    q = deq_feat.size(0)
                    feature_2d = feature_flat.permute(0, 2, 1).reshape(n * h * w, d)
                    dot = torch.matmul(feature_2d, deq_feat.T) # (n*h*w, q)
                    dot_exp = torch.exp(dot / args.temperature)

                    # pos, neg masks
                    with torch.no_grad():
                        if args.use_confident_region:
                            feature_2d_label_onehot = F.one_hot(pix_feat_label, 256)[..., :args.num_classes].float() # (n*h*w, c)
                        else:
                            feature_2d_label_onehot = F.one_hot(seed_from_sp.view(-1), 256)[..., :args.num_classes].float() # (n*h*w, c)
                        deq_label_onehot = F.one_hot(deq_label, args.num_classes).float() # (q, c)

                        pos_mask = torch.matmul(feature_2d_label_onehot, deq_label_onehot.T) # (n*h*w, q)
                        neg_mask = (1 - pos_mask) * feature_2d_label_onehot.max(1, keepdim=True)[0]

                        # select pos pairs by some strategy
                        #if args.random_pos:
                        if True:
                            rank_pos_mask = (torch.rand_like(pos_mask) + 1) * pos_mask
                        else:
                            rank_pos_mask = dot_exp * pos_mask
                        pos_mask = F.one_hot(rank_pos_mask.argmax(1), q).float() * pos_mask

                        # select neg pairs by some strategy
                        if args.hard_example_mining:
                            assert args.num_neg > 0 or args.ratio_neg > 0, (args.num_neg, args.ratio_neg)
                            rank_neg_mask = (dot - dot.min(1, keepdim=True)[0]) * neg_mask
                            if args.num_neg > 0:
                                topk = min(args.num_neg, q)
                                if args.numpy_hdem:
                                    rank_neg_mask_np = rank_neg_mask.data.cpu().numpy()
                                    topk_indices = rank_neg_mask_np.argsort(axis=1)[:, -topk:]
                                    neg_mask = F.one_hot(torch.from_numpy(topk_indices).to(gpu), q).max(1)[0].to(torch.float32) * neg_mask
                                else:
                                    _, rank_neg_mask_topk = torch.topk(rank_neg_mask, topk, dim=1, sorted=False)
                                    neg_mask = F.one_hot(rank_neg_mask_topk, q).to(torch.float32).sum(1)[0] * neg_mask # (n*h*w, q)
                            else:
                                max_neg_sim = rank_neg_mask.max(1, keepdim=True)[0]
                                th_neg_sim = max_neg_sim * args.ratio_neg
                                neg_mask = (rank_neg_mask > th_neg_sim).to(torch.float32) * neg_mask

                    logit_mem = dot_exp / torch.clamp_min( (dot_exp * (pos_mask + neg_mask)).sum(1, keepdim=True), 1e-5 )

                    if args.weight_pos:
                        with torch.no_grad():
                            #tmp_weight = probs_flat.max(1)[0].view(-1, 1)
                            tmp_weight = - dot.max(1)[0]
                            tmp_weight = tmp_weight - tmp_weight.min()
                            tmp_weight = tmp_weight / torch.clamp_min(tmp_weight.max(), 1e-5)
                            if args.weight_pos_pow != 1:
                                tmp_weight = tmp_weight**args.weight_pos_pow
                            pos_mask = pos_mask * tmp_weight.unsqueeze(1)

                    loss_mem = - ( torch.log(torch.clamp_min(logit_mem, 1e-5)) * pos_mask ).sum() / torch.clamp_min(pos_mask.sum(), 1e-5)
                    loss_mems.append(loss_mem)
                if len(loss_mems) == 1:
                    loss_mem = loss_mems[0]
                else:
                    if args.merge_mem_type == 'sum':
                        loss_mem = sum(loss_mems)
                    elif args.merge_mem_type == 'mean':
                        loss_mem = sum(loss_mems) / len(loss_mems)
                    else:
                        raise RuntimeError(args.merge_mem_type)
            else:
                loss_mem = feature_flat.sum() * 0

            # backward
            upper_iter = num_batch * 3.
            rampup = min((num_batch * epoch + batch) / upper_iter, 1.0)
            rampup = np.exp((rampup - 1) * 5)
            lambda_spce = args.lambda_spce * rampup
            lambda_mem = args.lambda_mem * rampup

            loss = loss_ce * args.lambda_ce + loss_spce * lambda_spce + loss_mem * lambda_mem
            loss.backward()
            optimizer.step()

            # monitor
            if is_log and (batch % args.log_frequency == 0):
                # loss
                meter.put('loss_ce', loss_ce.item())
                meter.put('loss_spce', loss_spce.item())
                if has_mem:
                    meter.put('loss_mem', loss_mem.item())
                    meter.put('nPos', pos_mask.sum().item()/(n * h * w))
                    meter.put('nNeg', neg_mask.sum().item()/(n * h * w))

                # miou
                with torch.no_grad():
                    # seg-CE
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

                    # CL
                    if has_mem:
                        dot_most_sim_val, dot_most_sim_id = dot.view(n, h, w, q)[vid].max(2) # (h, w)
                        dot_most_sim_cls = deq_label[dot_most_sim_id.view(-1)].view(h, w)

                # visualization 
                v_img = denormalize_image(image[vid].data.cpu().numpy().transpose(1, 2, 0)[..., ::-1])
                v_real_gt = get_segment_map(real_gt[vid])
                v_gt = get_segment_map(gt[vid])
                v_pred = get_segment_map(pred[vid])
                v_seed_from_sp = get_segment_map(seed_from_sp[vid].data.cpu().numpy())

                v_pix_feat_label = get_segment_map(pix_feat_label.view(n, h, w)[vid].data.cpu().numpy())
                v_sp_feat_label = sp_feat_label.view(n, args.max_superpixel)[vid][superpixel[vid]].view(h, w)
                v_sp_feat_label = get_segment_map(v_sp_feat_label.data.cpu().numpy())
                v_img_feat_label = get_segment_map(some_label.view(n, h, w)[vid].data.cpu().numpy())

                v_sp = superpixel_view[vid].data.cpu().numpy()
                v_sp = _SP_PALETTE[v_sp.ravel()].reshape(v_sp.shape+(3,))

                examples_ = [v_img, v_sp, v_real_gt, v_pred, v_seed_from_sp, v_gt, v_pix_feat_label, v_sp_feat_label, v_img_feat_label]
                #examples_ = [v_img, v_real_gt, v_gt, v_pred, v_seed_from_sp]
            
                #
                if has_mem:
                    v_dot_most_sim_val = get_score_map(dot_most_sim_val.data.cpu().numpy(), v_img)
                    v_dot_most_sim_cls = get_segment_map(dot_most_sim_cls.data.cpu().numpy())
                    examples_ += [v_dot_most_sim_val, v_dot_most_sim_cls]

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
    if _USE_PROJ_MODEL:
        mod = ProjectionModel(base="DeeplabV2_VGG16", base_args={
            "num_classes": 21,
            "pretrained": "./pretrained/vgg16_20M.pth"
        },
        base_stage_names=["c7"],
        base_stage_dims=[1024],
        proj_mid_dims=512,
        proj_out_dims=256,
        normalize=False
        ).to(gpu)
    else:
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

    dirname += '_' + args.embedding
    dirname += '_' + args.concat_layers.replace(',', '+')

    dirname += '_T{}'.format(args.temperature)
    dirname += '_spTh{}'.format(args.sp_label_threshold)
    dirname += '_capQ{}'.format(args.capacity)
    dirname += '_' + args.mem_type.replace(',', '+')
    dirname += '_lmCE{}_lmSPCE{}_lmMEM{}'.format(args.lambda_ce, args.lambda_spce, args.lambda_mem)
    if args.hard_example_mining:
        dirname += '_hdem{}'.format(args.ratio_neg if args.num_neg == 0 else args.num_neg)

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
            '/home/junsong_fan/diskf/data/VOC2012/superpixel/mcg_png')
            #'/home/junsong_fan/diskf/data/VOC2012/extra/Oversegment')

    parser.add_argument('--model', type=str, default='vgg16_largefov')
    parser.add_argument('--train-size', type=str, default='321,321')
    parser.add_argument('--test-scales', type=str, default='0.75,1,1.25')
    parser.add_argument('--num-classes', type=int, default=21, help='classification branch num_classes')

    parser.add_argument('--num-epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=16)

    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--sgd_mom', type=float, default=0.9)
    parser.add_argument('--sgd_wd', type=float, default=5e-4)

    parser.add_argument('--snapshot', type=str, default='./snapshot/ablation/voc')
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
    #parser.add_argument('--l2norm-multi', type=float, default=3.8)
    parser.add_argument('--temperature', type=float, default=0.1)
    
    parser.add_argument('--capacity', type=str, default='64')
    parser.add_argument('--mem-type', type=str, default='img')

    parser.add_argument('--lambda-ce', type=float, default=1)
    parser.add_argument('--lambda-spce', type=float, default=1)
    parser.add_argument('--lambda-mem', type=float, default=1)

    parser.add_argument('--only-infer', action='store_true')

    parser.add_argument('--random-pos', action='store_true')
    parser.add_argument('--separate', action='store_true')
    parser.add_argument('--merge-mem-type', type=str, default='sum')

    parser.add_argument('--hard-example-mining', action='store_true')
    parser.add_argument('--num-neg', type=int, default=0)
    parser.add_argument('--ratio-neg', type=float, default=0)
    parser.add_argument('--numpy-hdem', action='store_true')

    parser.add_argument('--weight-pos', action='store_true')
    parser.add_argument('--weight-pos-pow', type=float, default=1.0)
    #parser.add_argument('--align-hierarchies', action='store_true')

    #parser.add_argument('--num-point-mean', action='store_true')

    parser.add_argument('--use-confident-region', action='store_true')
    parser.add_argument('--use-intersection', action='store_true')

    args = parser.parse_args()

    if not args.only_infer:
        run(args, 'train')

    run(args, 'test')

