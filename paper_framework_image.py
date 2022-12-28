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

def main(gpu, args):
    is_distributed = False

    dataset = VOCPointDataset(
            args.test_image_root,
            args.
            None, args.test_gt_root, 'val',
            target_size=None, return_src=True)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False) if is_distributed else None
    loader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=False, drop_last=False,
            sampler=sampler, num_workers=1)

    # model
    mod = eval(args.model)(args.num_classes, dilation=12, upsample=args.upsample,
            embedding=args.embedding, concat_layers=args.concat_layers.split(',')).to(gpu)
    mod.set_mode('seg_embed')

    if args.force_pretrained:
        pretrained = args.force_pretrained
    else:
        pretrained = os.path.join(args.snapshot, 'checkpoint', '%s-%04d.pth' % (args.model, args.num_epochs-1))
    pretrained = torch.load(pretrained, map_location={'cuda:0': 'cuda:%s'%gpu})
    if is_distributed:
        mod = torch.nn.parallel.DistributedDataParallel(mod, device_ids=[gpu], find_unused_parameters=True)
    else:
        pretrained = {k.replace("module.", ""): v for k, v in pretrained.items()}
    mod.load_state_dict(pretrained, strict=True)
    mod.train(False)

    for image, label, src in loader:
        image = image.to(gpu)
        _, _, h, w = image.size()
        logit_seg, feature = mod(image.to(gpu))

