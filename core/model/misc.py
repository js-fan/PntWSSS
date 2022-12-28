import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from ..utils import denormalize_image
import pydensecrf.densecrf as dcrf

import cv2
import numpy as np


'''
Input:
    cam:   [N x C x H x W]
    image: [N x 3 x H x W], float32, normalized
    label: [N x C]
Output:
    out:   [N x H x W], int64, [0-(C+1), 255]
'''
__all__ = ['pseudo_label_from_cam', 'PseudoLabelFunction', 'PseudoLabel', 'crf_seed']

# crf
# def crf_seed(seed, image, fg_confidence=0.9, bg_confidence=0.6, crf_threshold=0.9, num_iter=5, normalized_image=True):
# crf2
def crf_seed(seed, image, fg_confidence=0.7, bg_confidence=0.6, crf_threshold=0.9, num_iter=5, normalized_image=True):
    device = None
    if isinstance(seed, torch.Tensor):
        device = seed.device
        seed = seed.data.cpu().numpy()
        image = image.data.cpu().numpy()

    if seed.ndim > 2:
        seed_output = np.array([crf_seed(seed_, image_, fg_confidence, bg_confidence, crf_threshold, num_iter, normalized_image) for seed_, image_ in zip(seed, image)])
    else:
        h, w = seed.shape
        unique_labels = [L for L in np.unique(seed) if L != 255]
        if len(unique_labels) < 2:
            return seed

        if image.shape[0] == 3:
            image = image.transpose(1, 2, 0)
        assert image.shape[2] == 3, image.shape

        if normalized_image:
            image = denormalize_image(image[..., ::-1])
        _h, _w = image.shape[:2]
        if (_h != h) or (_w != w):
            image = cv2.resize(image, (w, h))

        N = len(unique_labels)
        lookup_map = np.full((256, N), 1./N, np.float32)
        for i, L in enumerate(unique_labels):
            this_conf = bg_confidence if L == 0 else fg_confidence
            res_conf = (1 - this_conf) / (N - 1)
            vec_conf = np.full((1, N), res_conf, np.float32)
            vec_conf[0, i] = this_conf
            lookup_map[L] = vec_conf

        prob = np.ascontiguousarray(lookup_map[seed.ravel()].reshape(h * w, N).T)
        u = - np.log(np.maximum(prob, 1e-5))

        d = dcrf.DenseCRF2D(w, h, N)
        d.setUnaryEnergy(u)
        d.addPairwiseGaussian(sxy=3, compat=3)
        d.addPairwiseBilateral(sxy=50, srgb=5, rgbim=np.ascontiguousarray(image), compat=10)

        out_prob = np.array(d.inference(num_iter)).reshape(N, h, w)
        #out_prob = u
        seed_output = out_prob.argmax(axis=0)
        seed_output = np.array(unique_labels)[seed_output.ravel()].reshape(h, w)
        seed_output[out_prob.max(axis=0) < crf_threshold] = 255

    if device is not None:
        seed_output = torch.from_numpy(seed_output).to(device)

    return seed_output

class PseudoLabelFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cam, image, label, method):
        outputs = pseudo_label_from_cam(cam, image, label, method)
        return outputs

    @staticmethod
    def backward(ctx, output):
        return None, None, None, None

class PseudoLabel(torch.nn.Module):
    def __init__(self, method):
        assert isinstance(method, str), type(method)
        self.method = method

    def forward(self, cam, image, label):
        return PseudoLabelFunction.apply(cam, image, label, self.method)
    

def crf_inference(image, prob, confidence=None, num_iter=10):
    n, h, w = prob.shape
    assert image.dtype == np.uint8, image.dtype
    assert prob.shape[1:] == (h, w), (image.shape, prob.shape)

    if confidence is None:
        u = - np.log(np.maximum(prob.reshape(n, h * w), 1e-5))
    else:
        res_conf = (1. - confidence) / (n - 1)
        # unary
        u = np.full((n, h * w), res_conf, np.float32)
        prob_argmax = prob.argmax(axis=0).ravel()
        u[prob_argmax, np.arange(h * w)] = confidence
        prob_undefined = prob.max(axis=0).ravel() < 1
        u[:, prob_undefined] = 1. / n
        u = - np.log(np.maximum(u, 1e-5))
    
    d = dcrf.DenseCRF2D(w, h, n)
    d.setUnaryEnergy(u)
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=50, srgb=5, rgbim=np.ascontiguousarray(image), compat=10)

    out_prob = d.inference(num_iter)
    out_prob = np.array(out_prob).reshape(n, h, w)
    return out_prob

def make_irn_style_seed(img, cam, label):
    assert cam.shape[0] == len(label), (cam.shape, label)
    h, w = img.shape[:2]

    cam = np.maximum(cam, 0)
    cam = cam / np.maximum(cam.max(axis=(1, 2), keepdims=True), 1e-5)

    def _make_th_prob(th):
        inputs_with_bg = np.vstack([np.full((1, h, w), th, np.float32), cam]).reshape(-1, h * w)
        outputs = np.zeros_like(inputs_with_bg)
        outputs[inputs_with_bg.argmax(axis=0), np.arange(h * w)] = 1
        outputs = outputs.reshape(-1, h, w)
        return outputs

    fg_th, bg_th = 0.3, 0.05

    cam_fg = _make_th_prob(fg_th)
    crf_fg = crf_inference(img, cam_fg, 0.7).argmax(axis=0)

    cam_bg = _make_th_prob(bg_th)
    crf_bg = crf_inference(img, cam_bg, 0.7).argmax(axis=0)

    label_map = np.hstack([0, np.array(label) + 1])
    crf_out = label_map[crf_fg.ravel()].reshape(h, w).astype(np.uint8)
    crf_out[crf_fg == 0] = 255
    crf_out[np.maximum(crf_fg, crf_bg) == 0] = 0
    return crf_out

def make_naive_style_seed(img, cam, label):
    assert cam.shape[0] == len(label), (cam.shape, label)
    h, w = img.shape[:2]

    # cam, fg
    cam = np.maximum(cam, 0)
    cam = cam / np.maximum(cam.max(axis=(1, 2), keepdims=True), 1e-5)

    #fg_th, bg_th = 0.3, 0.05
    th =0.3

    # cam, bg
    bg_gamma = 4
    cam_bg = 1 - np.max(cam, axis=0, keepdims=True)
    if bg_gamma != 1:
        cam_bg = cam_bg**bg_gamma

    # cam, all
    cam_all = np.vstack([cam_bg, cam])
    sd_all = cam_all.argmax(axis=0)

    label_map = np.hstack([0, np.array(label) + 1])
    sd_out = label_map[sd_all.ravel()].reshape(h, w).astype(np.uint8)
    sd_out[cam_all.max(axis=0) < th] = 255
    return sd_out
    
def make_rrm_style_seed(img, cam, label):
    assert cam.shape[0] == len(label), (cam.shape, label)
    h, w = img.shape[:2]

    # cam, fg
    cam = np.maximum(cam, 0)
    cam = cam / np.maximum(cam.max(axis=(1, 2), keepdims=True), 1e-5)

    # seed-crf
    cam_bg = 1 - np.max(cam, axis=0, keepdims=True)

    cam_la = np.vstack([cam, cam_bg**4])
    crf_la = crf_inference(img, cam_la).argmax(axis=0)

    cam_ha = np.vstack([cam, cam_bg**32])
    crf_ha = crf_inference(img, cam_ha).argmax(axis=0)

    seed_crf = crf_la
    seed_crf[crf_la == 0] = 255
    seed_crf[crf_ha == 0] = 0

    # seed-cam
    cam_all = cam_ha
    cam_argmax = cam_all.argmax(axis=0)
    cam_sure_region = np.zeros((h, w), np.bool)
    for i, cam_i in enumerate(cam_all):
        if i == 0:
            # bg
            not_cam_argmax = cam_argmax != i
            cam_i[not_cam_argmax] = 0
            cam_sure_region ^= cam_i > 0.8
        else:
            # fg
            not_cam_argmax = cam_argmax != i
            cam_i[not_cam_argmax] = 0
            cam_i_order = cam_i[cam_i > 0.1]
            if cam_i_order.size == 0:
                continue
            cam_i_order = np.sort(cam_i_order)
            th = cam_i_order[int(cam_i_order.size * 0.6)]
            cam_sure_region ^= cam_i > th

    seed_crf[~cam_sure_region] = 255

    label_map = np.hstack([0, np.array(label) + 1])
    label_map = np.arange(256)
    for i, L in enumerate(label, 1):
        label_map[i] = L + 1
    seed_crf = label_map[seed_crf.ravel()].reshape(h, w).astype(np.uint8)
    return seed_crf

def pseudo_label_from_cam(cam, image, label, method='irn'):
    assert method in ['irn', 'naive', 'rrm'], method
    method_fn = eval('make_' + method + '_style_seed')

    N, C, H, W = cam.size()
    #assert image.size() == (N, 3, H, W), (image.size(), cam.size())
    assert image.size()[:2] == (N, 3), (image.size(), cam.size())
    assert label.size() == (N, C), (label.size(), cam.size())

    cam_ = cam.data.cpu().numpy()
    image_ = image.data.cpu().numpy()
    label_ = label.data.cpu().numpy()

    outputs = []
    for i in range(N):
        img_i = denormalize_image(image_[i].transpose(1, 2, 0)[..., ::-1])
        h, w = img_i.shape[:2]
        if (h != H) or (w != W):
            img_i = cv2.resize(img_i, (W, H))
        label_i = np.nonzero(label_[i])[0]
        cam_i = cam_[i][label_i]

        #outputs.append(make_irn_style_seed(img_i, cam_i, label_i))
        outputs.append(method_fn(img_i, cam_i, label_i))
    outputs = torch.from_numpy(np.array(outputs).astype(np.int64)).to(cam.device)
    return outputs

