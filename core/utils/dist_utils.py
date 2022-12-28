import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
from torch.utils.data import DataLoader

import os

def _get_free_port():
    # TODO: find free port.
    port = np.random.randint(2048, 65536)
    return port

def _start_process(rank, gpus, port, world_size, fn, args, kwargs):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    os.environ['MASTER_ADDR'] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    fn(*args, **kwargs)
    dist.barrier(device_ids=[rank])
    dist.destroy_process_group()

class Runner(object):
    def __init__(self, gpus, fn=None):
        if isinstance(gpus, int):
            gpus = str(gpus)
        elif isinstance(gpus, (list, tuple)):
            gpus = ",".join([str(gpu) for gpu in gpus])
        self.gpus = gpus
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus
        self.world_size = len(self.gpus.split(','))
        self.port = _get_free_port()
        self.fn = fn

    def run(self, *args, **kwargs):
        if self.world_size > 1:
            mp.spawn(_start_process,
                    args=(
                        self.gpus,
                        self.port,
                        self.world_size,
                        self.fn,
                        args,
                        kwargs
                        ),
                    nprocs=self.world_size,
                    join=True
                    )
        else:
            return self.fn(*args, **kwargs)

def get_dataloader(dataset, batch_size, shuffle=True, drop_last=True, num_workers=16):
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    if world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    else:
        sampler = None
    bs = batch_size // world_size
    loader = DataLoader(dataset, batch_size=bs, shuffle=shuffle and (sampler is None), pin_memory=False,
            drop_last=drop_last, sampler=sampler, num_workers=(num_workers+world_size-1) // world_size)
    return loader, sampler

def get_dist_info():
    is_dist = dist.is_initialized()
    rank = dist.get_rank() if is_dist else 0
    world_size = dist.get_world_size() if is_dist else 1
    return is_dist, rank, world_size

