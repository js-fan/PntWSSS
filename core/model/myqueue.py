import torch

class FeatureQueue(object):
    def __init__(self, num_classes, capacity):
        self.num_classes = num_classes
        self.capacity = capacity
        self.data = [None for _ in range(self.num_classes)]
    '''
    Input:
        feature:    [N-D-H-W] float Tensor
        class_id:   [N-H-W] int Tensor
        mask:       [N-H-W] bool Tensor
    '''
    def enqueue(self, feature, class_id, mask):
        feature = feature.permute(0, 2, 3, 1)
        for i in range(self.num_classes):
            mask_i = (class_id == i) & mask
            feature_i = feature[mask_i]
            if feature_i.size(0) > 0:
                self.put(i, feature_i)
        #print(feature.device, [x.size(0) if x is not None else 0 for x in self.data])
    '''
    Input:
        key:    int
        value:  [M-D] float Tensor
    '''
    def put(self, key, value):
        if value.size(0) > self.capacity:
            order = torch.randperm(value.size(0), device=value.device)[:self.capacity]
            value = value[order]
        if self.data[key] is None:
            self.data[key] = value
        else:
            self.data[key] = torch.cat([value, self.data[key]], dim=0)[:self.capacity]

    def get(self, key, num=None):
        if num is None:
            num = self.capacity
        return self.data[key][:num]

    def dequeue(self, num=None):
        if num is None:
            num = self.capacity
        min_length = min([0 if x is None else x.size(0) for x in self.data])
        num = min(min_length, num)
        if num == 0:
            return None

        data = torch.cat([x[:num].unsqueeze(0) for x in self.data], 0)
        return data

class SimpleQueue(object):
    def __init__(self, capacity):
        self.data = None
        self.dim = None
        self.capacity = capacity

    def put(self, data):
        if not isinstance(data, torch.Tensor):
            data = torch.cat(data, 0)
        size = data.size()
        assert len(size) == 2, 'Input should be 2d, but {}d is given.'.format(len(size))

        if self.data is None:
            self.data = data[:self.capacity]
            self.dim = size[1]
            return
        n, d = size
        assert d == self.dim, 'Dim not match: given {}, expected {}.'.format(d, self.dim)
        
        self.data = torch.cat([self.data, data], 0)
        if self.data.size(0) > self.capacity:
            self.data = self.data[-self.capacity:]

    def get(self):
        return self.data

    def size(self):
        return 0 if self.data is None else self.data.size(0)

class ClasswiseQueue(object):
    def __init__(self, capacity, num_classes, timestamp=False):
        self.data = [SimpleQueue(capacity) for L in range(num_classes)]
        self.capacity = capacity
        self.num_classes = num_classes
        self.timestamp = [SimpleQueue(capacity) for L in range(num_classes)] if timestamp else None
        self.curr_time = 0

    def put(self, data, label, ignore_index=255):
        # data:     (n, d), features
        # label:    (n,),   class label ids
        assert len(data.size()) == 2, data.size()
        assert len(label.size()) == 1, label.size()
        assert data.size(0) == label.size(0), (data.size(), label.size())

        unique_labels = torch.unique(label).data.cpu().numpy()
        for L in unique_labels:
            if L == ignore_index:
                continue
            #if L not in self.data:
            #    self.data[L] = SimpleQueue(self.capacity)
            data_L = data[label == L]
            self.data[L].put(data_L)
            if self.timestamp is not None:
                self.timestamp[L].put(torch.full((data_L.size(0), 1), self.curr_time, dtype=torch.int64, device=data_L.device))

        self.curr_time += 1

    def get(self):
        data = [self.data[L].get() for L in range(self.num_classes)]
        label = [torch.full((_data.size(0),), L, dtype=torch.int64, device=_data.device) \
                for L, _data in enumerate(data) if _data is not None]

        if len(label) == 0:
            return_vals = [None, None]
        else:
            data = torch.cat([_data for _data in data if _data is not None], dim=0)
            label = torch.cat(label, dim=0)
            return_vals = [data, label]

        if self.timestamp is not None:
            if return_vals[0] is None:
                timestamp = None
            else:
                timestamp = [self.timestamp[L].get() for L in range(self.num_classes)]
                timestamp = torch.cat([_time for _time in timestamp if _time is not None], dim=0).squeeze(1)
            return_vals.append(timestamp)
        return return_vals

    def size(self):
        return [self.data[L].size() for L in range(self.num_classes)]

