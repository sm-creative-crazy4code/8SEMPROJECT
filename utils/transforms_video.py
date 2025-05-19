"""This module contains transforms for videos."""

import numbers
import random

import numpy as np
from torchvision.transforms import RandomResizedCrop

from . import functional_video as F

__all__ = [
    "RandomResizedCropVideo",
    "CenterCropVideo",
    "NormalizeVideo",
    "ToTensorVideo",
    "RandomHorizontalFlipVideo",
]


class ResizeVideo:
    def __init__(self, size, interpolation_mode="bilinear"):
        self.size = size
        self.interpolation_mode = interpolation_mode

    def __call__(self, clip):
        return F.resize(clip, self.size, self.interpolation_mode)


class RandomResizedCropVideo(RandomResizedCrop):
    def __init__(
        self,
        size,
        crop,
        interpolation_mode="bilinear",
    ):
        if isinstance(size, tuple):
            assert len(size) == 2, "size should be tuple (height, width)"
            self.size = size
        else:
            self.size = (size, size)

        self.interpolation_mode = interpolation_mode
        self.crop = crop

    def __call__(self, clip):
       
        clip = F.resize(clip, self.size, self.interpolation_mode)
        # print(clip.shape)
        if clip.shape[2] - self.crop > 0:
            i = np.random.randint(clip.shape[2] - self.crop)
        else:
            i = 0
        if clip.shape[3] - self.crop > 0:
            j = np.random.randint(clip.shape[3] - self.crop)
        else:
            j = 0
        clip = clip[..., i : i + self.crop, j : j + self.crop]
        return clip

    def __repr__(self):
        return (
            self.__class__.__name__
            + f"(size={self.size}, interpolation_mode={self.interpolation_mode}, "
            + f"scale={self.scale}, ratio={self.ratio})"
        )


class CenterCropVideo:
    def __init__(self, crop_size):
        if isinstance(crop_size, numbers.Number):
            self.crop_size = (int(crop_size), int(crop_size))
        else:
            self.crop_size = crop_size

    def __call__(self, clip):
        
        return F.center_crop(clip, self.crop_size)

    def __repr__(self):
        return self.__class__.__name__ + f"(crop_size={self.crop_size})"


class NormalizeVideo:
   

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, clip):
       
        return F.normalize(clip, self.mean, self.std, self.inplace)

    def __repr__(self):
        return (
            self.__class__.__name__
            + f"(mean={self.mean}, std={self.std}, inplace={self.inplace})"
        )


class ToTensorVideo:
    

    def __init__(self):
        pass

    def __call__(self, clip):
        
        return F.to_tensor(clip)

    def __repr__(self):
        return self.__class__.__name__


class RandomHorizontalFlipVideo:
   
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, clip):
       
        if random.random() < self.p:
            clip = F.hflip(clip)
        return clip

    def __repr__(self):
        return self.__class__.__name__ + f"(p={self.p})"
