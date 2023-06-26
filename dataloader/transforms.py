from __future__ import division

import collections
import math
import numbers
import random
import types
import warnings

import cv2
import numpy as np
from PIL import Image, ImageOps



class CropCenterSquare(object):
    def __init__(self):
        pass

    def __call__(self, img):

        # assert img.width == mask.width
        # assert img.height == mask.height
        img_w, img_h = img.size
        h = min(img_h, img_w)
        crop = CenterCrop(h)

        return crop(img)


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):

        # assert img.width == mask.width
        # assert img.height == mask.height
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        # y1 = int(round((h - th) / 2.))
        y1 = int(round((h - th) / 2.))
        img = img.crop((x1, y1, x1 + tw, y1 + th))

        return img


class RandomRotation(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        return img.rotate(rotate_degree, Image.BILINEAR)

class RandomHorizontalFlip(object):
    def __call__(self, img):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


class RandomVerticalFlip(object):
    """Vertically flip the given PIL Image randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """
    def __call__(self, img):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        return img


class adjust_light():
    def __call__(self, image):
        seed = random.random()
        if seed > 0.5:
            gamma = random.random() * 3 + 0.5
            invGamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
            image = cv2.LUT(np.array(image).astype(np.uint8), table).astype(np.uint8)

        return image


class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)

        w, h = img.size
        ch, cw = self.size
        if w == cw and h == ch:
            return img
        if w < cw or h < ch:
            pw = cw - w if cw > w else 0
            ph = ch - h if ch > h else 0
            padding = (pw,ph,pw,ph)
            img  = ImageOps.expand(img,  padding, fill=0)
            w, h = img.size
            
        x1 = random.randint(0, w - cw)
        y1 = random.randint(0, h - ch)
        return img.crop((x1, y1, x1 + cw, y1 + ch))

class RandomScaleCrop(object):
    def __init__(self, size):
        self.size = size
        self.crop = RandomCrop(self.size)
            
    def __call__(self, img):

        #r = random.uniform(0.5, 2.0)
        w, h = img.size
        new_size = (int(w*random.uniform(1, 1.5)),int(h*random.uniform(1, 1.5)))
        return self.crop(img.resize(new_size, Image.BILINEAR))

