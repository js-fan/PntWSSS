import numpy as np
import os

class _ADE_proto(object):
    def __init__(self):
        curr_path = os.path.dirname(os.path.abspath(__file__))
        colors = np.load(os.path.join(curr_path, 'color150.npy'))
        self.palette = np.full((256, 3), 255, np.uint8)
        for i, c in enumerate(colors):
            self.palette[i] = c[::-1].astype(np.uint8)

ADE = _ADE_proto()
