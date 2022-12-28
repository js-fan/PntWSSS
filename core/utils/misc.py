import numpy as np
import time
import os
import logging
import datetime
from subprocess import call
from types import ModuleType
from collections import OrderedDict
import torch
from pathlib import Path
from pprint import pprint, pformat

def setGPU(gpus):
    len_gpus = len(gpus.split(','))
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    gpus = ','.join(map(str, range(len_gpus)))
    return gpus

def getTime():
    return datetime.datetime.now().strftime('%m-%d %H:%M:%S')

class Timer(object):
    def __init__(self):
        self.curr_record = time.time()
        self.prev_record = time.time()

    def record(self):
        self.prev_record = self.curr_record
        self.curr_record = time.time()

    def interval(self):
        if self.prev_record is None:
            return 0
        return self.curr_record - self.prev_record

def wrapColor(string, color):
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

def info(logger, msg, color=None):
    msg = '[{}]'.format(getTime()) + msg
    if logger is not None:
        logger.info(msg)

    if color is not None:
        msg = wrapColor(msg, color)
    print(msg)

def summaryArgs(logger, args, color=None):
    args = vars(args)
    keys = [key for key in args.keys() if key[:2] != '__']
    keys.sort()
    length = max([len(x) for x in keys])
    msg = [('{:<'+str(length)+'}: {}').format(k, args[k]) for k in keys]

    msg = '\n' + '\n'.join(msg)
    info(logger, msg, color)

class AvgMeter(object):
    def __init__(self, precision=4):
        self.data = {}
        self._prec = precision

    def put(self, key, value):
        return self.push(key, value)

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
            self.num_data = {}
        elif isinstance(key, str):
            del self.num_data[key]
        else:
            [self.clear(_key) for _key in key]

    def __str__(self):
        res = []
        for key, value in self.data.items():
            res.append(f"{key}={value[0]/value[1]:.{self._prec}f}")
        return ', '.join(res)

class SimpleETA(object):
    def __init__(self, total=None):
        self.init(total)

    def init(self, total):
        self.total = total
        self.cnt = 0
        self.tic = datetime.datetime.now()

    def update(self, cnt=1):
        self.cnt += cnt
        self.curr = datetime.datetime.now()

    def eta(self):
        eta = (self.curr - self.tic) * (max(self.total - self.cnt, 0.) / self.cnt)
        return str(eta).rsplit('.')[0]

class VersatileLogger(object):
    def __init__(self, logger):
        self.logger = logger
        self.meter = AvgMeter()
        self.eta = SimpleETA()

    def info(self, msg, color=None, *args, **kwargs):
        msg = f"{datetime.datetime.now().strftime('[%D %H:%M:%S]')} {msg}"
        msg_c = msg if color is None else wrapColor(msg, color)
        print(msg_c)
        return self.logger.info(msg, *args, **kwargs)

    def log_config(self, msg_dict, color=None):
        msg = 'CONFIG:\n' + pformat(msg_dict)
        self.info(msg, color)

    def flush(self, extra_msg=None, color=None):
        msgs = [str(self.meter), f"eta={self.eta.eta()}"]
        msgs.insert(0, extra_msg)
        msg = ', '.join(msgs)
        self.info(msg, color)

    def push(self, key, value):
        self.meter.push(key, value)
    
    def pull(self, key):
        self.meter.pull(key)
    
    def clear(self, key=None):
        self.meter.clear(key)

    def set_total(self, total):
        self.eta.init(total)

    def update(self, cnt=1):
        self.eta.update(cnt)


def get_logger(filename):
    Path(filename).parents[0].mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=filename, level=logging.INFO)
    logger = logging.getLogger("default_logger")
    return VersatileLogger(logger)

def getLogger(snapshot, model_name):
    if not os.path.exists(snapshot):
        os.makedirs(snapshot)
    logging.basicConfig(filename=os.path.join(snapshot, model_name+'.log'), level=logging.INFO)
    logger = logging.getLogger("monitorLogger")
    return logger

class SaveParams(object):
    def __init__(self, model, snapshot, model_name, num_save=5):
        self.model = model
        self.snapshot = snapshot
        self.model_name = model_name
        self.num_save = num_save
        self.cache_files = []

    def save(self, epoch):
        filename = os.path.join(self.snapshot, 'checkpoint', '{}-{:04d}.pth'.format(self.model_name, epoch))
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        torch.save(self.model.state_dict(), filename)
        self.cache_files.append(filename)

        while len(self.cache_files) > self.num_save:
            call(['rm', self.cache_files[0]])
            self.cache_files = self.cache_files[1:]

    def __call__(self, epoch):
        return self.save(epoch)

    @property
    def filename(self):
        return '' if len(self.cache_files) == 0 else self.cache_files[-1]

class LrScheduler(object):
    def __init__(self, method, init_lr, kwargs):
        self.method = method
        self.init_lr = init_lr

        if method == 'step':
            self.step_list = kwargs['step_list']
            self.factor = kwargs['factor']
            self.get = self._step
        elif method == 'poly':
            self.num_epochs = kwargs['num_epochs']
            self.power = kwargs['power']
            self.get = self._poly
        elif method == 'ramp':
            self.ramp_up = kwargs['ramp_up']
            self.ramp_down = kwargs['ramp_down']
            self.num_epochs = kwargs['num_epochs']
            self.scale = kwargs['scale']
            self.get = self._ramp
        else:
            raise ValueError(method)

    def _step(self, current_epoch):
        lr = self.init_lr
        step_list = [x for x in self.step_list]
        while len(step_list) > 0 and current_epoch >= step_list[0]:
            lr *= self.factor
            del step_list[0]
        return lr

    def _poly(self, current_epoch):
        lr = self.init_lr * ((1. - float(current_epoch)/self.num_epochs)**self.power)
        return lr

    def _ramp(self, current_epoch):
        if current_epoch < self.ramp_up:
            decay = np.exp(-(1 - float(current_epoch)/self.ramp_up)**2 * self.scale)
        elif current_epoch > (self.num_epochs - self.ramp_down):
            decay = np.exp(-(float(current_epoch+self.ramp_down-self.num_epochs)/self.ramp_down)**2 * self.scale)
        else:
            decay = 1.
        lr = self.init_lr * decay
        return lr

