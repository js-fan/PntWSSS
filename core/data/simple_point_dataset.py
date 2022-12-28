from .transforms import *
from pathlib import Path
from numbers import Number
from fast_slic.avx2 import SlicAvx2

def _as_list(x):
    return [x] if not isinstance(x, list) else x

class SimplePointDataset(torch.utils.data.Dataset):
    def __init__(self,
        data_roots,
        data_type,
        image_size,
        rand_short=False,
        rand_scale=False,
        rand_crop=False,
        rand_mirror=False,
        rand_point_shift=0,
        point_size=1,
        image_suffix=".png",
        max_superpixel=1024,
        post_resize_labels=None,
        online_superpixel=None,
        rtn_image_label=False,
        rtn_src=False,
        name_prefix=None
    ):
        data_roots = _as_list(data_roots)
        data_type = _as_list(data_type)
        assert len(data_roots) == len(data_type)

        non_point_data = [i for i, x in enumerate(data_type) if x != "Point"]
        self._nonPntIdx = non_point_data[0]
        self._point_data_indices = [i for i in range(len(data_type)) if i not in non_point_data]
        image_suffix = _as_list(image_suffix)
        if len(image_suffix) == 1:
            image_suffix = image_suffix * len(non_point_data)
        assert len(image_suffix) == len(non_point_data), (image_suffix, non_point_data)

        self.data = []
        self.data_wrapper = []

        for droot, dtype in zip(data_roots, data_type):
            wrapper = eval(f"Transformable{dtype}")
            self.data_wrapper.append(wrapper)

            if dtype == "Point":
                self._prepare_points(droot)
            else:
                suffix = image_suffix[0]
                image_suffix = image_suffix[1:]
                self._prepare_images(droot, suffix, name_prefix)

        self._len_data = len(self.data[0])
        assert all([len(data) == self._len_data for data in self.data]), ([len(x) for x in self.data])

        # rand_short OR rand_scale
        if rand_short:
            assert not rand_scale
        elif rand_scale:
            assert not rand_short
        self.image_size = image_size
        self.rand_short = rand_short
        self.rand_scale = rand_scale
        self.rand_crop = rand_crop
        self.rand_mirror = rand_mirror
        self.rand_point_shift = rand_point_shift
        self.point_size = point_size

        self.max_superpixel = max_superpixel
        if post_resize_labels is not None:
            if not isinstance(post_resize_labels, (list, tuple)):
                post_resize_labels = [int(post_resize_labels*x) for x in image_size]
            assert len(post_resize_labels) == 2, post_resize_labels
        self.post_resize_labels = post_resize_labels
        self.online_superpixel = online_superpixel
        self.rtn_src = rtn_src
        self.rtn_image_label = rtn_image_label
        self._resolve_image_labels()

    def _prepare_points(self, filename):
        with open(filename) as f:
            points = [x.strip().split(' ') for x in f.readlines()]
        _mapInt = lambda x: [int(float(_val)) for _val in x.split(',')]
        points = [pnt[0:1] + list(map(_mapInt, pnt[1:])) for pnt in points]
        self.data.append(sorted(points, key=lambda x: x[0]))

    def _prepare_images(self, dirname, suffix, name_prefix=None):
        images = [x.resolve() for x in Path(dirname).rglob(f"*{suffix}")]
        if name_prefix is not None:
            with open(name_prefix) as f:
                names = [x.strip() for x in f.readlines()]
            names = sorted(names)
            images = sorted(images, key=lambda x: x.name)
            match_images = []
            idx = 0
            for name in names:
                while not images[idx].name.startswith(name):
                    idx += 1
                match_images.append(images[idx])
                idx += 1
            assert len(match_images) == len(names), (len(match_images), len(names))
            images = match_images
        self.data.append(sorted(images, key=lambda x: x.name))

    def _resolve_image_labels(self):
        if not self.rtn_image_label:
            return
        assert len(self._point_data_indices) > 0
        # Assume using the first set of point labels
        points = self.data[self._point_data_indices[0]]
        image_labels = []
        num_classes = 0
        for pnt in points:
            label = [lxy[0] for lxy in pnt[1:]]
            image_labels.append(label)
            num_classes = max(num_classes, max(label))
        num_classes += 1
        self.num_classes = num_classes
        self.image_labels = image_labels

    def __len__(self):
        return self._len_data

    def __getitem__(self, index):
        items = []
        for data, wrapper in zip(self.data, self.data_wrapper):
            item = wrapper.load(data[index])
            items.append(item)

        output = self._transform(items)
        if self.rtn_image_label:
            label = torch.zeros((self.num_classes,), dtype=torch.int64, device='cpu')
            label[self.image_labels[index]] = 1
            output.append(label)

        if self.rtn_src:
            output.append(str(self.data[self._nonPntIdx][index]))

        if len(output) == 1:
            output = output[0]
        return output

    def _transform(self, items):
        # Get the image raw size, [H, W]
        H, W = items[self._nonPntIdx].shape[:2]

        # 1. Scale
        if self.rand_short:
            short = np.random.randint(self.rand_short[0], self.rand_short[1])
            s = float(short) / min(H, W)
            [item.scale(s) for item in items] 
            H, W = items[self._nonPntIdx].shape[:2]
        elif self.rand_scale:
            s = np.random.rand() * (self.rand_scale[1]-self.rand_scale[0]) + self.rand_scale[0]
            [item.scale(s) for item in items]
            H, W = items[self._nonPntIdx].shape[:2]

        # 2.1 Pad
        if self.image_size is None:
            h, w = H, W
        else:
            h, w = self.image_size
        ph, pw = max(h - H, 0), max(w - W, 0)
        if (ph > 0) or (pw > 0):
            tblr = [ph//2, ph-ph//2, pw//2, pw-pw//2]
            [item.pad(*tblr) for item in items]
            H, W = items[self._nonPntIdx].shape[:2]
        # 2.2 Crop
        ch, cw = H - h, W - w
        if self.rand_crop:
            tl = [np.random.randint(0, ch+1), np.random.randint(0, cw+1)]
            tblr = [tl[0], tl[0]+h, tl[1], tl[1]+w]
            [item.crop(*tblr) for item in items]
        
        # 3. Mirror
        for i in self._point_data_indices:
            items[i].size = [w, h]
        if self.rand_mirror and np.random.randint(2):
            [item.mirror() for item in items]

        # 4. [opt] Resize labels
        if self.post_resize_labels:
            size_ = self.post_resize_labels[::-1]
            for item in items:
                if isinstance(item, (
                    TransformablePoint,
                    TransformableSuperpixel,
                    TransformableLabel2D
                    )):
                    item.resize(size_)

        output = []
        for item in items:
            if isinstance(item, TransformablePoint):
                item = item.embed(self.rand_point_shift, self.point_size)
            elif isinstance(item, TransformableSuperpixel):
                item.rearrange(self.max_superpixel, 2)
            elif isinstance(item, TransformableImage):
                item.normalize()
                item.transpose(2, 0, 1)
            output.append(item.to_torch())

        if self.online_superpixel is not None:
            item = items[self.online_superpixel]
            output.insert(
                    self.online_superpixel+1,
                    self.get_online_superpixel(item)
            )
        return output
    
    def get_online_superpixel(self, item):
        assert isinstance(item, TransformableImage), type(item)
        img = TransformableImage(item.data.transpose(1, 2, 0)).denormalize().data.copy()
        slic = SlicAvx2(num_components=self.max_superpixel, compactness=10).iterate(img)
        return torch.from_numpy(slic.astype(np.int64))

    @classmethod
    def denormalize(cls, image):
        if isinstance(image, torch.Tensor):
            image = image.data.cpu().numpy()

        if (image.ndim == 2):
            return (image * 255.99).astype(np.uint8)

        assert image.ndim == 3, image.shape
        if image.shape[-1] != 3:
            assert image.shape[0] == 3, image.shape
            image = image.transpose(1, 2, 0)
        return TransformableImage(image).denormalize().get()
