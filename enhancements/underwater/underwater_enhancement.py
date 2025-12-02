#!/usr/bin/env python3

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple
from cvpods.data.registry import TRANSFORMS
from cvpods.data.transforms.transform_gen import TransformGen

@TRANSFORMS.register()
class UnderwaterColorCorrection(TransformGen):
    
    def __init__(self, alpha_r=1.3, alpha_g=1.0, alpha_b=0.7, **kwargs):
        self.alpha_r = alpha_r
        self.alpha_g = alpha_g
        self.alpha_b = alpha_b
    
    def __call__(self, image: np.ndarray, annotations=None, **kwargs) -> tuple:
        if len(image.shape) != 3 or image.shape[2] != 3:
            return image, annotations
        
        b, g, r = cv2.split(image.astype(np.float32))
        
        r_corrected = np.clip(r * self.alpha_r, 0, 255)
        g_corrected = np.clip(g * self.alpha_g, 0, 255)
        b_corrected = np.clip(b * self.alpha_b, 0, 255)
        
        corrected = cv2.merge([b_corrected, g_corrected, r_corrected])
        return corrected.astype(np.uint8), annotations

    def get_transform(self, img, annotations=None):
        return self

    def enable_record(self, mode=False):
        pass

@TRANSFORMS.register()
class UnderwaterContrastEnhancement(TransformGen):
    
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8), blend_ratio=0.5, **kwargs):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.blend_ratio = blend_ratio
        self.clahe = cv2.createCLAHE(
            clipLimit=clip_limit, 
            tileGridSize=tile_grid_size
        )
    
    def __call__(self, image: np.ndarray, annotations=None, **kwargs) -> tuple:
        if len(image.shape) != 3:
            return image, annotations
        
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        l_enhanced = self.clahe.apply(l)
        
        l_blended = cv2.addWeighted(l, self.blend_ratio, l_enhanced, 1 - self.blend_ratio, 0)
        
        lab_enhanced = cv2.merge([l_blended, a, b])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced, annotations

    def get_transform(self, img, annotations=None):
        return self

    def enable_record(self, mode=False):
        pass

@TRANSFORMS.register()
class UnderwaterDehazing(TransformGen):
    
    def __init__(self, omega=0.7, t0=0.2, radius=5, blend_ratio=0.3, **kwargs):
        self.omega = omega
        self.t0 = t0
        self.radius = radius
        self.blend_ratio = blend_ratio
    
    def dark_channel(self, image: np.ndarray, size: int = 15) -> np.ndarray:
        b, g, r = cv2.split(image)
        dark = cv2.min(cv2.min(r, g), b)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
        dark = cv2.erode(dark, kernel)
        return dark
    
    def atmospheric_light(self, image: np.ndarray, dark: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        img_size = h * w
        num_pixels = int(max(img_size / 1000, 1))
        
        dark_vec = dark.reshape(img_size)
        img_vec = image.reshape(img_size, 3)
        
        indices = dark_vec.argsort()[-num_pixels:]
        atmospheric_light = np.mean(img_vec[indices], axis=0)
        return atmospheric_light
    
    def __call__(self, image: np.ndarray, annotations=None, **kwargs) -> tuple:
        if len(image.shape) != 3:
            return image, annotations
        
        image_float = image.astype(np.float64) / 255.0
        
        dark = self.dark_channel(image_float)
        
        A = self.atmospheric_light(image_float, dark)
        
        transmission = 1 - self.omega * self.dark_channel(image_float / A)
        transmission = np.maximum(transmission, self.t0)
        
        result = np.zeros_like(image_float)
        for i in range(3):
            result[:, :, i] = (image_float[:, :, i] - A[i]) / transmission + A[i]
        
        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        
        result = cv2.addWeighted(image, self.blend_ratio, result, 1 - self.blend_ratio, 0)
        
        return result, annotations

    def get_transform(self, img, annotations=None):
        return self

    def enable_record(self, mode=False):
        pass

@TRANSFORMS.register()
class UnderwaterIlluminationCorrection(TransformGen):
    
    def __init__(self, gamma=1.5, **kwargs):
        self.gamma = gamma
    
    def __call__(self, image: np.ndarray, annotations=None, **kwargs) -> tuple:
        if len(image.shape) != 3:
            return image, annotations
        
        image_float = image.astype(np.float32) / 255.0
        corrected = np.power(image_float, 1.0 / self.gamma)
        corrected = np.clip(corrected * 255, 0, 255).astype(np.uint8)
        
        return corrected, annotations

    def get_transform(self, img, annotations=None):
        return self

    def enable_record(self, mode=False):
        pass


@TRANSFORMS.register()
class UnderwaterDegradationTransform(TransformGen):

    def __init__(self, prob=1.0, blur_kernel=9, color_shift=(0.8, 0.9, 1.0),
                 contrast_alpha=0.7, downscale_ratio=0.5, haze_coeff=0.4, **kwargs):
        self.prob = prob
        self.blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        self.color_shift = color_shift
        self.contrast_alpha = contrast_alpha
        self.downscale_ratio = downscale_ratio
        self.haze_coeff = haze_coeff

    def __call__(self, image, annotations=None, **kwargs):
        if np.random.random() > self.prob:
            return image, annotations

        if hasattr(image, 'mode'):
            im_np = np.array(image)
            if image.mode == 'RGB':
                im_np = cv2.cvtColor(im_np, cv2.COLOR_RGB2BGR)
        else:
            im_np = image.copy()

        h, w = im_np.shape[:2]

        b, g, r = cv2.split(im_np.astype(np.float32))
        r = r * self.color_shift[0]
        g = g * self.color_shift[1]
        b = b * self.color_shift[2]
        im_np = cv2.merge([b, g, r])

        if self.downscale_ratio > 0 and self.downscale_ratio < 1.0:
            small = cv2.resize(im_np, (int(w * self.downscale_ratio), int(h * self.downscale_ratio)), interpolation=cv2.INTER_LINEAR)
            im_np = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

        if self.blur_kernel > 1:
            im_np = cv2.GaussianBlur(im_np, (self.blur_kernel, self.blur_kernel), 0)

        im_np = im_np.astype(np.float32)
        im_np = (im_np - 127.5) * self.contrast_alpha + 127.5

        veil = np.ones_like(im_np) * 255.0
        im_np = im_np * (1.0 - self.haze_coeff) + veil * self.haze_coeff

        im_np = np.clip(im_np, 0, 255).astype(np.uint8)

        if hasattr(image, 'mode'):
            if image.mode == 'RGB':
                im_np = cv2.cvtColor(im_np, cv2.COLOR_BGR2RGB)
            from PIL import Image
            out = Image.fromarray(im_np)
        else:
            out = im_np

        return out, annotations

    def get_transform(self, img, annotations=None):
        return self

    def enable_record(self, mode=False):
        pass


class UnderwaterEnhancementModule:
    
    def __init__(self, 
                 enable_color_correction=True,
                 enable_contrast_enhancement=True, 
                 enable_dehazing=True,
                 enable_illumination_correction=True,
                 enhancement_weight=1.0,
                 color_alpha_r=1.3,
                 color_alpha_g=1.0,
                 color_alpha_b=0.7,
                 contrast_clip_limit=4.0,
                 gamma=1.3):
        self.enhancement_weight = enhancement_weight
        
        if enable_color_correction:
            self.color_corrector = UnderwaterColorCorrection()
        else:
            self.color_corrector = None
            
        if enable_contrast_enhancement:
            self.contrast_enhancer = UnderwaterContrastEnhancement()
        else:
            self.contrast_enhancer = None
            
        if enable_dehazing:
            self.dehazer = UnderwaterDehazing()
        else:
            self.dehazer = None
            
        if enable_illumination_correction:
            self.illumination_corrector = UnderwaterIlluminationCorrection()
        else:
            self.illumination_corrector = None
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        enhanced = image.copy()

        if self.color_corrector is not None:
            res = self.color_corrector(enhanced)
            enhanced = res[0] if isinstance(res, tuple) else res

        if self.contrast_enhancer is not None:
            res = self.contrast_enhancer(enhanced)
            enhanced = res[0] if isinstance(res, tuple) else res

        if self.dehazer is not None:
            res = self.dehazer(enhanced)
            enhanced = res[0] if isinstance(res, tuple) else res

        if self.illumination_corrector is not None:
            res = self.illumination_corrector(enhanced)
            enhanced = res[0] if isinstance(res, tuple) else res

        if self.enhancement_weight < 1.0:
            enhanced = cv2.addWeighted(
                image, 1 - self.enhancement_weight,
                enhanced, self.enhancement_weight, 0
            )

        return enhanced
    
    def enhance_batch(self, images: list) -> list:
        return [self(img) for img in images]

@TRANSFORMS.register()
class UnderwaterEnhancementTransform(TransformGen):
    
    def __init__(self, prob=0.8, **kwargs):
        self.prob = prob
        self.enhancer = UnderwaterEnhancementModule(**kwargs)
    
    def __call__(self, image, target=None, **kwargs):
        if np.random.random() < self.prob:
            if hasattr(image, 'mode'):
                image_np = np.array(image)
                if image.mode == 'RGB':
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            else:
                image_np = image
            
            enhanced_np = self.enhancer(image_np)
            
            if hasattr(image, 'mode'):
                if image.mode == 'RGB':
                    enhanced_np = cv2.cvtColor(enhanced_np, cv2.COLOR_BGR2RGB)
                from PIL import Image
                image = Image.fromarray(enhanced_np)
            else:
                image = enhanced_np
        
        return image, target

    def get_transform(self, img, annotations=None):
        return self

    def enable_record(self, mode=False):
        pass

__all__ = [
    'UnderwaterEnhancementModule',
    'UnderwaterEnhancementTransform',
    'UnderwaterColorCorrection',
    'UnderwaterContrastEnhancement', 
    'UnderwaterDehazing',
    'UnderwaterIlluminationCorrection'
]
