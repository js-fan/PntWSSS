import cv2
import os
import pickle
import numpy as np
from pathlib import Path

# bgr
meanf_list = [0.406, 0.456, 0.485]
meani_list = [104, 117, 124]
stdf_list  = [0.225, 0.224, 0.229]
_meanf = np.array(meanf_list).astype(np.float32)
_stdf = np.array(stdf_list).astype(np.float32)

def normalize_image(image):
    assert image.dtype == np.uint8, image.dtype
    return (image.astype(np.float32) / 255 - _meanf.reshape(1, 1, 3) ) / _stdf.reshape(1, 1, 3)

def denormalize_image(image):
    assert image.dtype in [np.float32, np.float64], image.dtype
    return np.clip( (image * _stdf.reshape(1, 1, 3) + _meanf.reshape(1, 1, 3)) * 255, a_min=0, a_max=255).astype(np.uint8)

def imwrite(filename, image):
    Path(filename).parents[0].mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(filename), image)

def npsave(filename, data):
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        try:
            os.makedirs(dirname)
        except:
            pass
    np.save(filename, data)

def pkldump(filename, data):
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        try:
            os.makedirs(dirname)
        except:
            pass
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def pklload(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def imhstack(images, height=None, thickness=3):
    images = as_list(images)
    images = list(map(image2C3, images))

    if height is None:
        height = np.array([img.shape[0] for img in images]).max()
    images = [resize_height(img, height) for img in images]

    if len(images) == 1:
        return images[0]

    if thickness > 0:
        images = [[img, np.full((height, thickness, 3), 255, np.uint8)] for img in images]
        images = np.hstack(sum(images, []))
    else:
        images = np.hstack(images)
    return images

def imvstack(images, width=None, thickness=3):
    images = as_list(images)
    images = list(map(image2C3, images))

    if width is None:
        width = np.array([img.shape[1] for img in images]).max()
    images = [resize_width(img, width) for img in images]

    if len(images) == 1:
        return images[0]

    if thickness > 0:
        images = [[img, np.full((thickness, width, 3), 255, np.uint8)] for img in images]
        images = np.vstack(sum(images, []))
    else:
        images = np.vstack(images)
    return images

def as_list(data):
    if not isinstance(data, (list, tuple)):
        return [data]
    return list(data)

def image2C3(image):
    if image.ndim == 3:
        return image
    if image.ndim == 2:
        return np.repeat(image[..., np.newaxis], 3, axis=2)
    raise ValueError("image.ndim = {}, invalid image.".format(image.ndim))

def resize_height(image, height):
    if image.shape[0] == height:
        return image
    h, w = image.shape[:2]
    width = height * w // h
    image = cv2.resize(image, (width, height))
    return image

def resize_width(image, width):
    if image.shape[1] == width:
        return image
    h, w = image.shape[:2]
    height = width * h // w
    image = cv2.resize(image, (width, height))
    return image

def imtext(image, text, space=(3, 3), color=(0, 0, 0), thickness=1, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.):
    assert isinstance(text, str), type(text)
    size = cv2.getTextSize(text, fontFace, fontScale, thickness)
    image = cv2.putText(image, text, (space[0], size[1]+space[1]), fontFace, fontScale, color, thickness)
    return image

def norm_score(score):
    if len(np.unique(score)) == 1:
        return np.ones_like(score) if score.max() > 0 else np.zeros_like(score)
    score = score - score.min()
    score /= max(score.max(), 1e-5)
    return score

def get_score_map(score, image=None, TYPE=cv2.COLORMAP_JET):
    score = norm_score(score)
    score = cv2.applyColorMap((score * 255.99).astype(np.uint8), TYPE)
    if image is not None:
        h, w = image.shape[:2]
        score = cv2.resize(score, (w, h))
        score = cv2.addWeighted(image, 0.2, score, 0.8, 0)
    return score

def patch_images(images, nRow, nCol, default_shape):
    if len(images) == 0:
        images = [np.full(default_shape, 255, np.uint8)]
    if len(images) < nRow * nCol:
        images += [np.full_like(images[-1], 255, np.uint8)] * (nRow * nCol - len(images))
    images = images[:nRow * nCol]
    images = imvstack([imhstack(images[i : i+nCol], height=240) for i in range(0, nRow*nCol, nCol)])
    return images

