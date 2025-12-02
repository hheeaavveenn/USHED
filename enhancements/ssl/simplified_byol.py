#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
import copy
try:
    from torch.cuda import amp
    _AMP_AVAILABLE = True
except Exception:
    _AMP_AVAILABLE = False


class SimpleBYOLHead(nn.Module):
    
    def __init__(self, in_dim=2048, proj_dim=256, hidden_dim=2048):
        super().__init__()
        
        self.projector = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim)
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(proj_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, proj_dim)
        )
        
    def forward(self, x, use_predictor=True):
        if len(x.shape) == 4:
            x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        
        z = self.projector(x)
        
        if use_predictor:
            p = self.predictor(z)
            return z, p
        else:
            return z


class EMAUpdater:
    
    def __init__(self, decay=0.996):
        self.decay = decay
    
    def update(self, online_net, target_net):
        for online_param, target_param in zip(online_net.parameters(), target_net.parameters()):
            target_param.data = self.decay * target_param.data + (1 - self.decay) * online_param.data


class USHEDBYOL(nn.Module):
    
    def __init__(self, ushed_model, byol_weight=0.5, ema_decay=0.996, 
                 feature_dim=None, proj_dim=256, detach_target=True):
        super().__init__()
        
        try:
            feature_dim = ushed_model.backbone.bottom_up.output_shape()['res5'].channels
        except Exception:
            feature_dim = ushed_model.backbone.output_shape().get('p5', list(ushed_model.backbone.output_shape().values())[-1]).channels
        
        self.ushed = ushed_model
        
        self.byol_weight = byol_weight
        self.detach_target = detach_target
        
        self.byol_head = SimpleBYOLHead(feature_dim, proj_dim)
        
        device = self.ushed.device
        self.byol_head.to(device)
        
        self.target_byol_head = copy.deepcopy(self.byol_head)
        
        self.target_byol_head.to(device)
        
        for param in self.target_byol_head.parameters():
            param.requires_grad = False
        
        self.ema_updater = EMAUpdater(ema_decay)
        
        self.training_step = 0
        
    def extract_backbone_features(self, images):
        if hasattr(self.ushed, 'device'):
            device = self.ushed.device
        else:
            device = next(self.ushed.parameters()).device
        
        if images.device != device:
            images = images.to(device)
            
        backbone = self.ushed.backbone
        try:
            bottom_up_feats = backbone.bottom_up(images)
            if isinstance(bottom_up_feats, dict) and 'res5' in bottom_up_feats:
                return bottom_up_feats['res5']
            if not isinstance(bottom_up_feats, dict):
                return bottom_up_feats[-1] if isinstance(bottom_up_feats, (list, tuple)) else bottom_up_feats
        except Exception:
            pass

        fpn_feats = backbone(images)
        if isinstance(fpn_feats, dict):
            if 'p5' in fpn_feats:
                return fpn_feats['p5']
            return list(fpn_feats.values())[-1]
        return fpn_feats
    
    def byol_loss(self, online_proj, online_pred, target_proj):
        online_pred = F.normalize(online_pred, dim=1)
        target_proj = F.normalize(target_proj, dim=1)
        
        loss = 2 - 2 * (online_pred * target_proj).sum(dim=1).mean()
        return loss
    
    def forward(self, batched_inputs, teacher_model=None):
        self.training_step += 1
        
        device = next(self.ushed.parameters()).device
        for batch_item in batched_inputs:
            for key, value in batch_item.items():
                if isinstance(value, torch.Tensor) and value.device != device:
                    batch_item[key] = value.to(device)
        
        ushed_outputs = self.ushed(batched_inputs, teacher_model=teacher_model)
        
        if self.training:
            loss_dict, custom_metrics_dict = ushed_outputs
            try:
                byol_loss = self.compute_byol_loss(batched_inputs)
                loss_dict["byol_loss"] = byol_loss * self.byol_weight
            except Exception as e:
                import traceback
                print(f\"BYOL loss computation failed: {e}\")
                traceback.print_exc()
                pass
            
            if self.training_step % 1 == 0:
                self.ema_updater.update(self.byol_head, self.target_byol_head)
            
            return loss_dict, custom_metrics_dict
        
        return ushed_outputs
    
    def compute_byol_loss(self, batched_inputs):
        images = self.extract_images_from_batch(batched_inputs)
        if len(images) == 0:
            return torch.tensor(0.0, device=next(self.ushed.parameters()).device)

        device = next(self.ushed.parameters()).device

        views1 = []
        views2 = []
        for img in images:
            img = img.to(device).float()
            img01 = img / 255.0 if img.max() > 1.0 else img
            v1 = self.light_augmentation(img01)
            v2 = self.strong_augmentation(img01)
            v1 = self._resize_short_edge(v1, short_edge=720)
            v2 = self._resize_short_edge(v2, short_edge=720)
            v1 = torch.clamp(v1 * 255.0, 0.0, 255.0)
            v2 = torch.clamp(v2 * 255.0, 0.0, 255.0)
            views1.append(v1)
            views2.append(v2)

        max_h = max(v.shape[1] for v in views1)
        max_w = max(v.shape[2] for v in views1)
        
        views1_resized = []
        views2_resized = []
        for v1, v2 in zip(views1, views2):
            v1_resized = F.interpolate(v1.unsqueeze(0), size=(max_h, max_w), mode='bilinear', align_corners=False).squeeze(0)
            v2_resized = F.interpolate(v2.unsqueeze(0), size=(max_h, max_w), mode='bilinear', align_corners=False).squeeze(0)
            views1_resized.append(v1_resized)
            views2_resized.append(v2_resized)
        
        view1_batch = self.ushed.normalizer(torch.stack(views1_resized, dim=0))
        view2_batch = self.ushed.normalizer(torch.stack(views2_resized, dim=0))

        autocast_ctx = amp.autocast(enabled=_AMP_AVAILABLE) if _AMP_AVAILABLE else torch.cuda.amp.autocast(enabled=False)
        with autocast_ctx:
            feat1 = self.extract_backbone_features(view1_batch)
            pooled1 = F.adaptive_avg_pool2d(feat1, (1, 1)).flatten(1)
            online_proj1, pred1 = self.byol_head(pooled1, use_predictor=True)
        with torch.no_grad():
            with autocast_ctx:
                target1 = self.target_byol_head(pooled1, use_predictor=False)
        del feat1, pooled1, online_proj1
        
        with autocast_ctx:
            feat2 = self.extract_backbone_features(view2_batch)
            pooled2 = F.adaptive_avg_pool2d(feat2, (1, 1)).flatten(1)
            online_proj2, pred2 = self.byol_head(pooled2, use_predictor=True)
        with torch.no_grad():
            with autocast_ctx:
                target2 = self.target_byol_head(pooled2, use_predictor=False)
        del feat2, pooled2, online_proj2

        loss1 = self.byol_loss(online_pred=pred1, target_proj=target2, online_proj=None)
        loss2 = self.byol_loss(online_pred=pred2, target_proj=target1, online_proj=None)
        del pred1, pred2, target1, target2
        return (loss1 + loss2) / 2.0
    
    def extract_images_from_batch(self, batched_inputs):
        images = []
        for batch_item in batched_inputs:
            if "image_Nor" in batch_item:
                img = batch_item["image_Nor"]
            elif "image" in batch_item:
                img = batch_item["image"]
            else:
                img_keys = [k for k in batch_item.keys() if k.startswith("image")]
                if img_keys:
                    img = batch_item[img_keys[0]]
                else:
                    raise KeyError("not")
            
            images.append(img)
        
        return images
    
    def create_augmented_views(self, images):
        view1_list = []
        view2_list = []
        
        for img in images:
            if img.dtype != torch.float32:
                img = img.float() / 255.0
            
            view1 = self.light_augmentation(img)
            view1_list.append(view1)
            
            view2 = self.strong_augmentation(img)
            view2_list.append(view2)
        
        return torch.stack(view1_list), torch.stack(view2_list)
    
    def light_augmentation(self, img):
        if torch.rand(1, device=img.device).item() > 0.5:
            img = torch.flip(img, dims=[2])
        
        if torch.rand(1, device=img.device).item() > 0.3:
            brightness_factor = (0.9 + torch.rand(1, device=img.device, dtype=img.dtype) * 0.2)
            img = img * brightness_factor
            img = torch.clamp(img, 0, 1)
        
        return img
    
    def strong_augmentation(self, img):
        if torch.rand(1, device=img.device).item() > 0.5:
            img = torch.flip(img, dims=[2])

        if torch.rand(1, device=img.device).item() > 0.2:
            contrast_factor = 0.6 + torch.rand(1, device=img.device, dtype=img.dtype) * 0.8
            mean = img.mean(dim=[1, 2], keepdim=True)
            img = (img - mean) * contrast_factor + mean

            gray = img.mean(dim=0, keepdim=True)
            saturation_factor = 0.4 + torch.rand(1, device=img.device, dtype=img.dtype) * 1.2
            img = gray + (img - gray) * saturation_factor
            img = torch.clamp(img, 0, 1)

        if torch.rand(1, device=img.device).item() < 0.3:
            gray = img.mean(dim=0, keepdim=True)
            img = gray.expand_as(img)

        if torch.rand(1, device=img.device).item() < 0.5:
            img = self._gaussian_blur(img, kernel_size=5, sigma=1.0 + torch.rand(1, device=img.device).item())

        if torch.rand(1, device=img.device).item() < 0.1:
            threshold = 0.5
            img = torch.where(img > threshold, 1.0 - img, img)

        if torch.rand(1, device=img.device).item() > 0.7:
            noise = torch.randn_like(img) * 0.02
            img = torch.clamp(img + noise, 0, 1)

        return img

    def _resize_short_edge(self, img, short_edge=720):
        c, h, w = img.shape
        short = min(h, w)
        if short <= short_edge:
            return img
        scale = float(short_edge) / float(short)
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))
        img = F.interpolate(img.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False).squeeze(0)
        return img

    def _gaussian_blur(self, img, kernel_size=5, sigma=1.0):
        import math
        c, h, w = img.shape
        radius = kernel_size // 2
        x = torch.arange(-radius, radius + 1, dtype=img.dtype, device=img.device)
        gauss = torch.exp(-(x ** 2) / (2 * sigma * sigma))
        gauss = gauss / gauss.sum()
        kernel1d = gauss.view(1, 1, -1)
        img_b = img.unsqueeze(0)
        weight_w = kernel1d.repeat(c, 1, 1)
        img_b = F.conv2d(img_b, weight_w.unsqueeze(2), padding=(0, radius), groups=c)
        weight_h = kernel1d.repeat(c, 1, 1)
        img_b = F.conv2d(img_b, weight_h.unsqueeze(3), padding=(radius, 0), groups=c)
        return img_b.squeeze(0)


def create_byol_enhanced_model(ushed_model, config_overrides=None):
    
    default_config = {
        'byol_weight': 0.3,
        'ema_decay': 0.996,
        'feature_dim': getattr(getattr(ushed_model.backbone, 'bottom_up', ushed_model.backbone), 'output_shape')().get('res5', None).channels
                      if hasattr(getattr(ushed_model.backbone, 'bottom_up', ushed_model.backbone), 'output_shape') and 
                         'res5' in getattr(ushed_model.backbone, 'bottom_up').output_shape() else
                      ushed_model.backbone.output_shape().get('p5', list(ushed_model.backbone.output_shape().values())[-1]).channels,
        'proj_dim': 256,
        'detach_target': True
    }
    
    if config_overrides:
        default_config.update(config_overrides)
    
    return USHEDBYOL(ushed_model, **default_config)


__all__ = [
    'SimpleBYOLHead',
    'USHEDBYOL', 
    'create_byol_enhanced_model'
]
