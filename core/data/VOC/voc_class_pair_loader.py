import numpy as np
import torch
from .voc_class_loader import VOCClassDataset, VOC, fn_load_transform

class VOCClassPairDataset(VOCClassDataset):
    def __init__(self, *args, **kwargs):
        super(VOCClassPairDataset, self).__init__(*args, **kwargs)

        assert self.data_labels is not None, "'label_root' required"
        self.data_labels_ = list(map(VOC.get_annotation, self.data_labels))
        self.should_len = len(self.data_images) // 2 * 2
        self.reset(0)

    def reset(self, seed):
        np.random.seed(seed)
        index_pool = list(np.random.permutation(len(self.data_images)))
        index_pool = index_pool + index_pool
        reorder_index = []
        check_pair = lambda x, y: len(set(x) - set(y)) < len(x)

        while len(reorder_index) < self.should_len:
            left = index_pool.pop()
            right = index_pool.pop()
            cache = []

            while not check_pair(self.data_labels_[left], self.data_labels_[right]):
                cache.append(right)
                right = index_pool.pop()
            reorder_index.append(left)
            reorder_index.append(right)
            for idx in cache[::-1]:
                index_pool.append(idx)

        self.order_index = reorder_index

    def __getitem__(self, index):
        idx = self.order_index[index]
        src = self.data_images[idx]
        img = fn_load_transform(src, self.target_size, self.rand_crop, self.rand_mirror, self.rand_scale)

        lbl = torch.zeros((20,), dtype=torch.int64)
        lbl[self.data_labels_[idx]] = 1

        if self.return_src:
            return img, lbl, src
        return img, lbl

    def __len__(self):
        return self.should_len

class DistributedPairSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size, num_replicas=None, rank=None, shuffle=True, seed=0, drop_last=False):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()

        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.epoch = 0
        self.drop_last = drop_last

        total_pairs = len(self.dataset) // 2
        num_batch_pairs = self.batch_size // 2
        block_size = self.num_replicas * num_batch_pairs
        if drop_last and total_pairs % block_size != 0:
            self.num_batch = int(np.ceil( (total_pairs - block_size) / block_size ))
        else:
            self.num_batch = int(np.ceil(total_pairs / block_size))
        self.num_pairs = self.num_batch * num_batch_pairs

        self.total_batch = self.num_batch * self.num_replicas
        self.total_pairs = self.total_batch * num_batch_pairs
        self.num_batch_pairs = num_batch_pairs
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            self.dataset.reset(self.seed + self.epoch)
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset) // 2, generator=g).tolist()  # type: ignore
            indices = list(map(lambda x: x * 2, indices))
        else:
            indices = list(np.arange(len(self.dataset) // 2) * 2)

        if not self.drop_last:
            padding_size = self.total_pairs - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * np.ceil(padding_size / len(indices)))[:padding_size]
        else:
            indices = indices[:self.total_pairs]
        assert len(indices) == self.total_pairs, (self.total_pairs, len(indices))

        # subsample
        indices = indices[self.rank:self.total_pairs:self.num_replicas]
        assert len(indices) == self.num_pairs, (self.num_pairs, len(indices))

        batch = []
        for idx in indices:
            batch.append(idx)
            if len(batch) == self.num_batch_pairs:
                batch = batch + [idx_ + 1 for idx_ in batch]
                yield batch
                batch = []

    def __len__(self) -> int:
        return self.num_batch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

