import warnings
import logging
import datetime
from pprint import pformat
from pathlib import Path
from subprocess import call
import cv2
from .image_utils import imvstack, imhstack

import numpy as np
import torch

def _get_base_logger(filename):
    filename = Path(filename)
    filename.parents[0].mkdir(parents=True, exist_ok=True)

    logging.basicConfig(filename=filename, level=logging.INFO)
    logger = logging.getLogger("default_logger")
    logger.info(f"Created logger at:{datetime.datetime.now().strftime('%D %H:%M:%S')}")
    if not filename.exists():
        warnings.warn("Failed to create logger with 'logging'")
        logger =_Navie_Logger_Emulator(filename)
    return logger

class _Navie_Logger_Emulator(object):
    def __init__(self, filename):
        self.filename = filename

    def info(self, msg):
        with open(self.filename, 'a') as f:
            f.write(msg + '\n')

class AvgMeter(object):
    def __init__(self, precision=4):
        self.data = {}
        self._prec = precision

    def push(self, key, value):
        if key not in self.data:
            self.data[key] = [0., 0.]
        self.data[key][0] += value
        self.data[key][1] += 1

    def pull(self, key):
        val, cnt = self.data[key]
        return val / cnt

    def clear(self, key=None):
        if key is None:
            self.data = {}
        elif isinstance(key, str):
            del self.data[key]
        else:
            [self.clear(_key) for _key in key]

    def __str__(self):
        res = []
        for key, value in self.data.items():
            res.append(f"{key}={value[0]/value[1]:.{self._prec}f}")
        return ', '.join(res)

class SimpleETA(object):
    def __init__(self, total=None):
        self.total = total
        self.cnt = 0
        self.tic = datetime.datetime.now()
        self.toc = None

    def step(self):
        self.cnt += 1
        self.toc = datetime.datetime.now()

    def eta(self):
        if self.toc is None:
            raise RuntimeError
        eta = (self.toc - self.tic) * (max(self.total - self.cnt, 0.) / self.cnt)
        return str(eta).rsplit('.')[0]

class ImageMonitor(object):
    def __init__(self, dirname):
        self.dirname = Path(dirname)
        self.dirname.mkdir(parents=True, exist_ok=True)
        self.images = []
        self.cnt = 0

    def clear(self):
        self.images = []

    def push(self, img):
        assert isinstance(img, np.ndarray), type(img)
        self.images.append(img)

    def flush_last(self):
        if len(self.images) > 0:
            cv2.imwrite(str(self.dirname / "last.jpg"), self.images[-1])

    def flush(self, name=None):
        name = str(self.cnt) if name is None else str(name)
        self.cnt += 1
        if len(self.images) > 0:
            image = imvstack(self.images)
            cv2.imwrite(str(self.dirname / f"{name}.jpg"), image)

def _warp_color(string, color):
    try:
        code = {
            'red': 91, 'r': 91,
            'green': 92, 'g': 92,
            'yellow': 93, 'y': 93,
            'blue': 94, 'b': 94,
            'purple': 95, 'p': 95,
            'cyan': 96, 'c': 96,
            'bold': 1,
            'underline': 4
        }[color.lower()]
    except KeyError:
        raise RuntimeError(f"Unknown color: {color}")
    return f"\033[{code}m{string}\033[0m"

class SimpleMonitor(object):
    def __init__(self, filename, total=None):
        self.logger = _get_base_logger(filename)
        self.meter = AvgMeter()
        self.imager = ImageMonitor(Path(filename).parents[0] / "trainImages")
        self.cnt = 0
        self.eta = self.init_eta(total)
        self.log_freq = None

    def init_eta(self, total):
        self.total = total
        self.eta = None if total is None else SimpleETA(total)

    def init_auto_log(self, log_freq, template=None):
        self.log_freq = log_freq
        self.auto_log_template = template

    @property
    def requires_log(self):
        return self.log_freq and (self.cnt % self.log_freq == 0)

    def clear(self):
        self.meter.clear()
        self.imager.clear()

    def step(self, *args):
        if self.eta is not None:
            self.eta.step()

        if self.requires_log:
            if (self.auto_log_template is None) or (len(args) == 0):
                extra_msg = None 
            else:
                extra_msg = self.auto_log_template.format(*args)
            self.flush(extra_msg)
        self.cnt += 1

    def info(self, msg, c=None, *args, **kwargs):
        msg = f"{datetime.datetime.now().strftime('[%D %H:%M:%S]')} {msg}"
        msg_c = msg if c is None else _warp_color(msg, c)
        print(msg_c)
        return self.logger.info(msg, *args, **kwargs)

    def log_config(self, cfg_dict, c=None):
        msg = "CONFIG:\n" + pformat(cfg_dict)
        self.info(msg, c)

    def flush(self, extra_msg=None, c=None):
        msgs = [f"Iter=[{self.cnt}{'' if self.total is None else '/'+str(self.total)}]", str(self.meter)]
        if self.eta is not None:
            msgs.append(f"eta={self.eta.eta()}")
        if extra_msg is not None:
            if isinstance(extra_msg, str):
                msgs.insert(0, extra_msg)
            elif isinstance(extra_msg, (list, tuple)):
                msgs = extra_msg + msgs
            else:
                raise RuntimeError(type(extra_msg))
        msg = ', '.join(msgs)
        self.info(msg, c)
        self.imager.flush_last()

    def flush_images(self, name=None):
        self.imager.flush(name)

    def push(self, k, v):
        return self.meter.push(k, v)

    def pull(self, k):
        return self.meter.pull(k)

    def push_image(self, img):
        return self.imager.push(img)


def get_monitor(filename):
    return SimpleMonitor(filename)

class Saver(object):
    def __init__(self, model, dirname, modname, k_recent=3):
        self.mod = model
        self.dirname = Path(dirname)
        self.name = modname
        self.k_recent = k_recent
        self.history = []
        self.dirname.mkdir(parents=True, exist_ok=True)

    def save(self, identifier):
        filename = self.dirname / f"{self.name}-{identifier}.pth"
        torch.save(self.mod.state_dict(), str(filename))
        self.history.append(filename)
        self.check_delete()

    def check_delete(self):
        if len(self.history) > self.k_recent:
            for filename in self.history[:-self.k_recent]:
                if not filename.exists(): continue
                call(['rm', str(filename)])
            self.history = self.history[-self.k_recent:]

    def __str__(self):
        if len(self.history) == 0:
            return ''
        return str(self.history[-1])
