from .cityscape_labels import labels
import numpy as np

class _CS_proto(object):
    def __init__(self):
        self.id2trainId = np.full((256,), 255, np.int32)
        for L in labels:
            if L.ignoreInEval:
                continue
            self.id2trainId[L.id] = L.trainId

        self.trainId2id = np.full((256,), 255, np.int32)
        for L in labels:
            if L.ignoreInEval:
                continue
            self.trainId2id[L.trainId] = L.id

        self.id2catId = np.full((256,), 255, np.int32)
        for L in labels:
            if L.ignoreInEval:
                continue
            self.id2catId[L.id] = L.categoryId - 1

        self.trainId2catId = np.full((256,), 255, np.int32)
        for L in labels:
            if L.ignoreInEval:
                continue
            self.trainId2catId[L.trainId] = L.categoryId - 1

        self.palette = np.full((256, 3), 200, np.uint8)
        for L in labels:
            if L.ignoreInEval:
                continue
            self.palette[L.trainId] = np.array(L.color).astype(np.uint8)[::-1]

        self.paletteId = np.full((257, 3), 200, np.uint8)
        for L in labels:
            self.paletteId[L.id] = np.array(L.color).astype(np.uint8)[::-1]

        self.trainId2name = {L.trainId : L.name for L in labels if not L.ignoreInEval}
        self.catId2name = {L.categoryId - 1 : L.category for L in labels if not L.ignoreInEval}

        self.name2trainId = {L.name : L.trainId for L in labels}
        self.name2catId = {L.category : L.categoryId - 1 for L in labels}
        self.name2id = {L.name : L.id for L in labels}

CS = _CS_proto()
