import logging
import math
from typing import List

import cv2
import json
import numpy as np
import os
import torch
import torch.nn.functional as F
from cvpods.data.build import build_dataset
from cvpods.layers import ShapeSpec, batched_nms, cat
from cvpods.modeling.box_regression import Shift2BoxTransform
from cvpods.modeling.losses import sigmoid_focal_loss_jit, iou_loss
from cvpods.modeling.postprocessing import detector_postprocess
from cvpods.structures import BoxMode
from cvpods.structures import Boxes, ImageList, Instances, pairwise_iou
from cvpods.utils import log_first_n
from pycocotools.coco import COCO
from torch import nn

from ushed.modeling.mpo import mpopp_weights
from ushed.modeling.dpf import DPFScheduler





def fp16_clamp(x, min=None, max=None):
    if not x.is_cuda and x.dtype == torch.float16:
        return x.float().clamp(min, max).half()

    return x.clamp(min, max)


def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6):

    assert mode in ['iou', 'iof', 'giou'], f'Unsupported mode {mode}'
    assert (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0)

    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new(batch_shape + (rows, ))
        else:
            return bboxes1.new(batch_shape + (rows, cols))

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (
        bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (
        bboxes2[..., 3] - bboxes2[..., 1])

    if is_aligned:
        lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])
        rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])

        wh = fp16_clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
    else:
        lt = torch.max(bboxes1[..., :, None, :2],
                       bboxes2[..., None, :, :2])
        rb = torch.min(bboxes1[..., :, None, 2:],
                       bboxes2[..., None, :, 2:])

        wh = fp16_clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1[..., None] + area2[..., None, :] - overlap
        else:
            union = area1[..., None]
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :, None, :2],
                                    bboxes2[..., None, :, :2])
            enclosed_rb = torch.max(bboxes1[..., :, None, 2:],
                                    bboxes2[..., None, :, 2:])

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union
    if mode in ['iou', 'iof']:
        return ious
    enclose_wh = fp16_clamp(enclosed_rb - enclosed_lt, min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    enclose_area = torch.max(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious

def Revision_GT(Instance1,gt_Instance2):
    assert Instance1.image_size == gt_Instance2.image_size
    image_size = Instance1.image_size

    refine_gt_Instance = Instances(tuple(image_size))
    missing_Instance = Instances(tuple(image_size))

    bboxes1 = Instance1.pred_boxes.tensor
    scores1 = Instance1.scores
    classes1 = Instance1.pred_classes

    bboxes2 = gt_Instance2.gt_boxes.tensor.clone()
    classes2 = gt_Instance2.gt_classes.clone()

    ious = bbox_overlaps(bboxes1,bboxes2)

    while(True):
        refine_gt_inds =  (ious>=0.5).any(dim=0)

        if torch.where(refine_gt_inds)[0].numel() == 0:
            break

        refine_inds = ious.max(dim=0)[1]

        refine_pred_classes = classes1[refine_inds]
        refine_pred_classes = refine_pred_classes == classes2

        diff_labels_inds = (~refine_gt_inds | ~refine_pred_classes) & refine_gt_inds

        diff_labels_inds0 = torch.where(diff_labels_inds)[0]

        if diff_labels_inds0.numel()>0:
            index = [refine_inds[diff_labels_inds],diff_labels_inds0]

            input_zeros = torch.zeros((diff_labels_inds0.numel())).to(ious.device)

            ious.index_put_(index,input_zeros)

        else:
            break

    if torch.where(refine_gt_inds)[0].numel() > 0:
        refine_inds = refine_inds[refine_gt_inds]
        refine_gt_inds_repeat = torch.where(refine_gt_inds)[0].reshape(-1,1).repeat(1,4)
        bboxes2.scatter_(dim=0,index=refine_gt_inds_repeat,src=bboxes1[refine_inds])

    refine_gt_Instance.gt_boxes = Boxes((bboxes2+bboxes2.abs())/2)
    refine_gt_Instance.gt_classes = classes2


    missing_inds = (ious<0.5).all(dim=1)
    missing_Instance.pred_boxes = Boxes(bboxes1[missing_inds])
    missing_Instance.pred_classes = classes1[missing_inds]
    missing_Instance.scores = scores1[missing_inds]

    return missing_Instance,refine_gt_Instance


def Revision_PRED(Instance1,Instance2):
    if Instance1.image_size != Instance2.image_size:
        image_size = (max(Instance1.image_size[0], Instance2.image_size[0]), 
                     max(Instance1.image_size[1], Instance2.image_size[1]))
    else:
        image_size = Instance1.image_size

    refine_gt_Instance = Instances(tuple(image_size))
    missing_Instance = Instances(tuple(image_size))

    bboxes1 = Instance1.pred_boxes.tensor
    scores1 = Instance1.scores
    classes1 = Instance1.pred_classes

    bboxes2 = Instance2.pred_boxes.tensor.clone()
    scores2 = Instance2.scores.clone()
    classes2 = Instance2.pred_classes.clone()

    ious = bbox_overlaps(bboxes1,bboxes2)

    if len(ious)==0 or len(ious[0])==0:
        if Instance1.image_size != Instance2.image_size:
            unified_size = (max(Instance1.image_size[0], Instance2.image_size[0]), 
                           max(Instance1.image_size[1], Instance2.image_size[1]))
            new_instance1 = Instances(unified_size)
            new_instance2 = Instances(unified_size)
            for field in Instance1.get_fields():
                setattr(new_instance1, field, getattr(Instance1, field))
            for field in Instance2.get_fields():
                setattr(new_instance2, field, getattr(Instance2, field))
            return Instances.cat([new_instance1, new_instance2])
        else:
            return Instances.cat([Instance1,Instance2])

    while(True):

        refine_gt_inds = (ious > 0.5).any(dim=0)
        refine_inds = ious.max(dim=0)[1]

        refine_pred_scores = scores1[refine_inds]
        need_refine = refine_pred_scores >= scores2

        lower_scores_inds = (~refine_gt_inds | ~need_refine) & refine_gt_inds

        lower_scores_inds0 = torch.where(lower_scores_inds)[0]

        if lower_scores_inds0.numel()>0:
            index = [refine_inds[lower_scores_inds],lower_scores_inds0]

            input_zeros = torch.zeros((lower_scores_inds0.numel())).to(ious.device) + 0.5

            ious.index_put_(index,input_zeros)

        else:
            break

    refine_inds = refine_inds[refine_gt_inds]
    refine_gt_inds = torch.where(refine_gt_inds)[0]
    refine_gt_inds_repeat = refine_gt_inds.reshape(-1,1).repeat(1,4)

    bboxes2.scatter_(dim=0,index=refine_gt_inds_repeat,src=bboxes1[refine_inds])
    classes2.scatter_(dim=0,index=refine_gt_inds,src=classes1[refine_inds])
    scores2.scatter_(dim=0,index=refine_gt_inds,src=scores1[refine_inds])

    refine_gt_Instance.pred_boxes = Boxes((bboxes2+bboxes2.abs())/2)
    refine_gt_Instance.pred_classes = classes2
    refine_gt_Instance.scores = scores2


    missing_inds = (ious<0.5).all(dim=1)
    missing_Instance.pred_boxes = Boxes(bboxes1[missing_inds])
    missing_Instance.pred_classes = classes1[missing_inds]
    missing_Instance.scores = scores1[missing_inds]

    if missing_Instance.image_size != refine_gt_Instance.image_size:
        unified_size = (max(missing_Instance.image_size[0], refine_gt_Instance.image_size[0]), 
                       max(missing_Instance.image_size[1], refine_gt_Instance.image_size[1]))
        missing_Instance = Instances(unified_size, **missing_Instance.get_fields())
        refine_gt_Instance = Instances(unified_size, **refine_gt_Instance.get_fields())
    
    return Instances.cat([missing_Instance,refine_gt_Instance])


def bbox2points(box):
    min_x, min_y, max_x, max_y = torch.split(box[:, :4], [1, 1, 1, 1], dim=1)

    return torch.cat(
                [min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y], dim=1
            ).reshape(
                -1, 2
            )

def points2bbox(point, max_w, max_h):
    point = point.reshape(-1, 4, 2)
    if point.size()[0] > 0:
        min_xy = point.min(dim=1)[0]
        max_xy = point.max(dim=1)[0]
        xmin = min_xy[:, 0].clamp(min=0, max=max_w)
        ymin = min_xy[:, 1].clamp(min=0, max=max_h)
        xmax = max_xy[:, 0].clamp(min=0, max=max_w)
        ymax = max_xy[:, 1].clamp(min=0, max=max_h)
        min_xy = torch.stack([xmin, ymin], dim=1)
        max_xy = torch.stack([xmax, ymax], dim=1)
        return torch.cat([min_xy, max_xy], dim=1)
    else:
        return point.new_zeros(0, 4)

def permute_to_N_HWA_K(tensor, K):
    assert tensor.dim() == 4, tensor.shape
    N, _, H, W = tensor.shape
    tensor = tensor.view(N, -1, K, H, W)
    tensor = tensor.permute(0, 3, 4, 1, 2)
    tensor = tensor.reshape(N, -1, K)
    return tensor


def permute_all_cls_and_box_to_N_HWA_K_and_concat_retina(box_cls,
                                                  box_delta,
                                                  num_classes=80):
    box_cls_flattened = [permute_to_N_HWA_K(x, num_classes) for x in box_cls]
    box_delta_flattened = [permute_to_N_HWA_K(x, 4) for x in box_delta]
    box_cls = cat(box_cls_flattened, dim=1).view(-1, num_classes)
    box_delta = cat(box_delta_flattened, dim=1).view(-1, 4)
    return box_cls, box_delta


def permute_all_cls_and_box_to_N_HWA_K_and_concat_fcos(box_cls,
                                                  box_delta,
                                                  box_center,
                                                  num_classes=80):
    box_cls_flattened = [permute_to_N_HWA_K(x, num_classes) for x in box_cls]
    box_delta_flattened = [permute_to_N_HWA_K(x, 4) for x in box_delta]
    box_center_flattened = [permute_to_N_HWA_K(x, 1) for x in box_center]
    box_cls = cat(box_cls_flattened, dim=1).view(-1, num_classes)
    box_delta = cat(box_delta_flattened, dim=1).view(-1, 4)
    box_center = cat(box_center_flattened, dim=1).view(-1, 1)
    return box_cls, box_delta, box_center


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


class FCOS(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.num_classes = cfg.MODEL.FCOS.NUM_CLASSES
        self.in_features = cfg.MODEL.FCOS.IN_FEATURES
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.focal_loss_alpha = cfg.MODEL.FCOS.FOCAL_LOSS_ALPHA
        self.focal_loss_gamma = cfg.MODEL.FCOS.FOCAL_LOSS_GAMMA
        self.iou_loss_type = cfg.MODEL.FCOS.IOU_LOSS_TYPE
        self.center_sampling_radius = cfg.MODEL.FCOS.CENTER_SAMPLING_RADIUS
        self.pseudo_score_thres = cfg.MODEL.FCOS.PSEUDO_SCORE_THRES
        self.matching_iou_thres = cfg.MODEL.FCOS.MATCHING_IOU_THRES
        self.start_splg_iter = cfg.TRAINER.get('START_SPLG_ITER', None)

        self.score_threshold = cfg.MODEL.FCOS.SCORE_THRESH_TEST
        self.topk_candidates = cfg.MODEL.FCOS.TOPK_CANDIDATES_TEST
        self.nms_threshold = cfg.MODEL.FCOS.NMS_THRESH_TEST
        self.max_detections_per_image = cfg.TEST.DETECTIONS_PER_IMAGE

        self.backbone = cfg.build_backbone(
            cfg, input_shape=ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN)))

        backbone_shape = self.backbone.output_shape()
        feature_shapes = [backbone_shape[f] for f in self.in_features]
        self.head = FCOSHead(cfg, feature_shapes)
        self.shift_generator = cfg.build_shift_generator(cfg, feature_shapes)

        self.shift2box_transform = Shift2BoxTransform(
            weights=cfg.MODEL.FCOS.BBOX_REG_WEIGHTS)
        self.object_sizes_of_interest = cfg.MODEL.FCOS.OBJECT_SIZES_OF_INTEREST

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(
            3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(
            3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.in_normalizer = lambda x: x * pixel_std + pixel_mean
        self.mpo_pp_cfg = getattr(cfg.MODEL, "MPO_PP", {}) or {}
        self.dp_cfg = getattr(cfg.MODEL.FCOS, "DYNAMIC_PSEUDO", {}) or {}
        self.max_iter = (
            int(getattr(cfg.SOLVER.LR_SCHEDULER, "MAX_ITER", 0))
            if hasattr(cfg, "SOLVER") and hasattr(cfg.SOLVER, "LR_SCHEDULER")
            else 0
        )
        self._dpf_scheduler = DPFScheduler(
            cfg=self.dp_cfg,
            default_score_thr=float(self.pseudo_score_thres),
            default_iou_thr=float(self.matching_iou_thres),
            max_iter=self.max_iter,
        )
        self.to(self.device)

    def _dpf_dynamic_score_thr(self, curr_iter: int) -> float:
        if hasattr(self, "_dpf_scheduler"):
            return float(self._dpf_scheduler.score_thr(curr_iter))
        return float(self.pseudo_score_thres)

    def _dpf_dynamic_iou_thr(self, curr_iter: int) -> float:
        if hasattr(self, "_dpf_scheduler"):
            return float(self._dpf_scheduler.iou_thr(curr_iter))
        return float(self.matching_iou_thres)


        
    def freeze_all(self):
        for module in self.modules():
            for key, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                value.requires_grad = False

   
    def forward(self, batched_inputs, teacher_model=None):
        images_raw, images_nor,images_str = self.preprocess_image(batched_inputs,teacher_model)
        
        if self.training:
            gt_instances_raw = [x["instances_Raw"].to(self.device) for x in batched_inputs]
            gt_instances_nor = [x["instances_Nor"].to(self.device) for x in batched_inputs]
            gt_instances_str = [x["instances_Str"].to(self.device) for x in batched_inputs]
            
            features_raw = teacher_model.backbone(images_raw.tensor)
            features_nor = self.backbone(images_nor.tensor) 
            features_str = self.backbone(images_str.tensor)

            features_raw = [features_raw[f] for f in self.in_features]
            features_nor = [features_nor[f] for f in self.in_features]
            features_str = [features_str[f] for f in self.in_features]

            shifts_raw = self.shift_generator(features_raw)
            shifts_nor = self.shift_generator(features_nor)
            shifts_str = self.shift_generator(features_str)

            TM_Raw = [x["transform_matrix_Raw"] for x in batched_inputs]
            TM_Nor = [x["transform_matrix_Nor"] for x in batched_inputs]
            TM_Str = [x["transform_matrix_Str"] for x in batched_inputs]

        else:
            gt_instances_raw = None
            gt_instances_nor = [x["instances"].to(self.device) for x in batched_inputs]
            gt_instances_str = None
            features_raw = None
            features_nor = self.backbone(images_nor.tensor)
            features_nor = [features_nor[f] for f in self.in_features]
            features_str = None
            shifts_raw = None
            shifts_nor = self.shift_generator(features_nor)
            shifts_str = None
            TM_Raw = None
            TM_Nor = None
            TM_Str = None
        
        cls, delta, center, cls_aug, delta_aug, center_aug = self.head(features_nor, features_str)
        
        if self.training:
            with torch.no_grad():
                curr_iter = batched_inputs[0].get('iter', 0) if isinstance(batched_inputs, list) and len(batched_inputs) > 0 else 0
                thr = self._dpf_dynamic_score_thr(int(curr_iter))
                iou_thr = self._dpf_dynamic_iou_thr(int(curr_iter))
                
                head1_preds = self.pseudo_gt_generate(
                    [ar.clone() for ar in cls],
                    [br.clone() for br in delta],
                    [cr.clone() for cr in center],
                    shifts_nor,
                    images_nor,
                    pseudo_score_thres=thr
                )
                head2_preds = self.pseudo_gt_generate(
                    [ar.clone() for ar in cls_aug],
                    [br.clone() for br in delta_aug],
                    [cr.clone() for cr in center_aug],
                    shifts_str,
                    images_str,
                    pseudo_score_thres=thr
                )

                if teacher_model:

                    missing_Instances,refine_gt_Instances,high_score_Instances = self.teacher_revision(
                                    teacher_model,features_raw,gt_instances_raw,shifts_raw,images_raw)

                    high_score_Instances_to_nor = self.cvt_Instance_list(TM_Raw,high_score_Instances,TM_Nor,images_nor.image_sizes)
                    high_score_Instances_to_str = self.cvt_Instance_list(TM_Raw,high_score_Instances,TM_Str,images_str.image_sizes)

                    head1_preds_new = [Revision_PRED(missing_Instance,head1_pred) for missing_Instance,head1_pred in zip(high_score_Instances_to_nor,head1_preds)]
                    head2_preds_new = [Revision_PRED(missing_Instance,head2_pred) for missing_Instance,head2_pred in zip(high_score_Instances_to_str,head2_preds)]
                    
                else:
                    head1_preds_new = head1_preds
                    head2_preds_new = head2_preds

                sawn_head1 = []
                sawn_head2 = []
                quality_total_iter = 0
                quality_kept_iter = 0
                for p1, p2, img_sz in zip(head1_preds_new, head2_preds_new, images_nor.image_sizes):
                    if p1.has('pred_boxes'):
                        boxes1 = p1.pred_boxes.tensor
                        scores1 = p1.scores
                        labels1 = p1.pred_classes
                    else:
                        sawn_head1.append(p1)
                        sawn_head2.append(p2)
                        continue
                    if p2.has('pred_boxes'):
                        boxes2 = p2.pred_boxes.tensor
                        scores2 = p2.scores
                        labels2 = p2.pred_classes
                    else:
                        boxes2 = boxes1.new_zeros((0,4)); scores2 = scores1.new_zeros((0,)); labels2 = labels1.new_zeros((0,), dtype=labels1.dtype)
                    _cfg_q = dict(self.mpo_pp_cfg)
                    _cfg_q['THR_REF'] = float(thr)
                    w1 = mpopp_weights(
                        preds_v1=dict(boxes=boxes1, scores=scores1, labels=labels1),
                        preds_v2=dict(boxes=boxes2, scores=scores2, labels=labels2),
                        feats_v1=None, feats_v2=None,
                        image_size_hw=img_sz, cfg=_cfg_q
                    )
                    q_thr = float(self.mpo_pp_cfg.get('QUALITY_THR', 0.5)) if isinstance(self.mpo_pp_cfg, dict) else 0.5
                    keep = w1['quality'] >= q_thr
                    try:
                        quality_total_iter += int(w1['quality'].numel())
                        quality_kept_iter += int(keep.long().sum().item())
                    except Exception:
                        pass
                    from cvpods.structures import Instances, Boxes
                    out1 = Instances(img_sz)
                    out1.pred_boxes = Boxes(w1['boxes'][keep])
                    out1.scores = w1['scores'][keep]
                    out1.pred_classes = w1['labels'][keep]
                    sawn_head1.append(out1)

                    if p2.has('pred_boxes') and p2.pred_boxes.tensor.numel() > 0:
                        _cfg_q2 = dict(self.mpo_pp_cfg)
                        _cfg_q2['THR_REF'] = float(thr)
                        w2 = mpopp_weights(
                            preds_v1=dict(boxes=boxes2, scores=scores2, labels=labels2),
                            preds_v2=dict(boxes=boxes1, scores=scores1, labels=labels1),
                            feats_v1=None, feats_v2=None,
                            image_size_hw=img_sz, cfg=_cfg_q2
                        )
                        q_thr2 = float(self.mpo_pp_cfg.get('QUALITY_THR', 0.5)) if isinstance(self.mpo_pp_cfg, dict) else 0.5
                        keep2 = w2['quality'] >= q_thr2
                        out2 = Instances(img_sz)
                        out2.pred_boxes = Boxes(w2['boxes'][keep2])
                        out2.scores = w2['scores'][keep2]
                        out2.pred_classes = w2['labels'][keep2]
                        sawn_head2.append(out2)
                    else:
                        sawn_head2.append(p2)

                head1_preds_new = sawn_head1
                head2_preds_new = sawn_head2

                gt_instances1 = self.merge_ground_truth(gt_instances_nor, head2_preds_new, images_nor, iou_thr,TM_Str,TM_Nor)
                gt_instances2 = self.merge_ground_truth(gt_instances_str, head1_preds_new, images_str, iou_thr,TM_Nor,TM_Str)

                gt_classes_fcos1, gt_shifts_reg_deltas1, gt_centerness1,valid_img_1 = self.get_fcos_ground_truth(
                    shifts_nor, gt_instances1)

                gt_classes_fcos2, gt_shifts_reg_deltas2, gt_centerness2,valid_img_2 = self.get_fcos_ground_truth(
                    shifts_str, gt_instances2)

            fcos_losses1 = self.fcos_losses1(gt_classes_fcos1, gt_shifts_reg_deltas1, gt_centerness1,
                                           cls, delta, center,valid_img_1)

            fcos_losses2 = self.fcos_losses2(gt_classes_fcos2, gt_shifts_reg_deltas2, gt_centerness2,
                                           cls_aug, delta_aug, center_aug,valid_img_2)

            costom_num = {
                'found_num': 0,
                'all_num': 0,
                'wrong_loc_num': 0,
                'ori_num': 0,
                'wrong_cls_num': 0,
                'with_SPLG': 0,
            }

            curr_iter = batched_inputs[0].get('iter', 0) if isinstance(batched_inputs, list) and len(batched_inputs) > 0 else 0
            splg_dpfive = (teacher_model is not None) and (self.start_splg_iter is not None) and (curr_iter >= self.start_splg_iter)

            if splg_dpfive:
                costom_num['with_SPLG'] = 1
                iou_thr_stats = self._dpf_dynamic_iou_thr(int(curr_iter))
                if teacher_model is not None:
                    for gt_i, pred_i in zip(gt_instances_nor, high_score_Instances_to_nor):
                        gt_boxes = gt_i.gt_boxes
                        gt_classes = gt_i.gt_classes
                        num_gt = len(gt_boxes)
                        costom_num['all_num'] += int(num_gt)
                        if num_gt == 0:
                            costom_num['wrong_loc_num'] += int(len(pred_i.pred_boxes))
                            continue

                        if len(pred_i.pred_boxes) == 0:
                            costom_num['ori_num'] += int(num_gt)
                            continue

                        iou_mat = pairwise_iou(gt_boxes, pred_i.pred_boxes)
                        covered_gt = torch.zeros((num_gt,), dtype=torch.bool, device=iou_mat.device)

                        best_gt_iou, best_gt_idx = iou_mat.max(dim=0)
                        for j in range(best_gt_iou.numel()):
                            iou_val = float(best_gt_iou[j].item())
                            if iou_val >= float(iou_thr_stats):
                                gt_idx = int(best_gt_idx[j].item())
                                pred_cls = int(pred_i.pred_classes[j].item())
                                gt_cls = int(gt_classes[gt_idx].item())
                                covered_gt[gt_idx] = True
                                if pred_cls == gt_cls:
                                    costom_num['found_num'] += 1
                                else:
                                    costom_num['wrong_cls_num'] += 1
                            else:
                                costom_num['wrong_loc_num'] += 1

                        costom_num['ori_num'] += int((~covered_gt).sum().item())

            loss_dict = dict(fcos_losses1, **fcos_losses2)

            return loss_dict, costom_num

        else:
            results = self.inference(cls, delta, center, shifts_nor, images_nor)

            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                    results, batched_inputs, images_nor.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    @torch.no_grad()
    def merge_ground_truth(self, targets, predictions, images, iou_thresold,source_TM,target_TM):

        new_targets = []

        for targets_per_image, predictions_per_image, image,source_TM_i,target_TM_i in zip(targets, predictions, images,source_TM,target_TM):
            image_size = image.shape[1:3]

            if len(predictions_per_image.pred_boxes) != 0:
                predictions_per_image_cvt = self.cvt_bbox(source_TM_i,
                                                            predictions_per_image.pred_boxes.tensor,
                                                            target_TM_i,
                                                            image.shape,
                                                            predictions_per_image.pred_classes,
                                                            predictions_per_image.scores
                                                            )
            else:
                predictions_per_image_cvt = predictions_per_image

            iou_matrix = pairwise_iou(targets_per_image.gt_boxes,
                                      predictions_per_image_cvt.pred_boxes)
            iou_filter = iou_matrix > iou_thresold

            target_class_list = (targets_per_image.gt_classes).reshape(-1, 1)
            pred_class_list = (predictions_per_image_cvt.pred_classes).reshape(1, -1)
            class_filter = target_class_list == pred_class_list

            final_filter = iou_filter & class_filter
            unlabel_idxs = torch.sum(final_filter, 0) == 0

            new_target = Instances(image_size)
            new_target.gt_boxes = Boxes.cat([targets_per_image.gt_boxes,
                                             predictions_per_image_cvt.pred_boxes[unlabel_idxs]])
            new_target.gt_classes = torch.cat([targets_per_image.gt_classes,
                                               predictions_per_image_cvt.pred_classes[unlabel_idxs]])
            new_targets.append(new_target)

        return new_targets





    def cvt_bbox(self,source_TM,source_bbox,target_TM,target_img_shape,labels,scores):
        
        source_points = bbox2points(source_bbox[:, :4])
        source_points = torch.cat(
                    [source_points, source_points.new_ones(source_points.shape[0], 1)], dim=1
                )
        M_T = np.matmul(target_TM,np.linalg.inv(source_TM))
        M_T = torch.tensor(M_T).to(source_bbox.device).float()
        target_points = (M_T @ source_points.t()).t()
        _,H,W = target_img_shape
        target_points = target_points[:, :2] / target_points[:, 2:3]
        target_bboxes = points2bbox(target_points, W, H)
        minx,min_y,max_x,max_y = target_bboxes[:,0],target_bboxes[:,1],target_bboxes[:,2],target_bboxes[:,3]
        valid_bboxes = (minx < max_x) & (min_y < max_y)

        target = Instances(target_img_shape[1:3])
        target.pred_boxes = Boxes(target_bboxes[valid_bboxes])
        target.scores = scores[valid_bboxes]
        target.pred_classes = labels[valid_bboxes]
        return target

    def cvt_Instance_list(self,source_TM,source_Instance,target_TM,target_img_shapes):
        assert len(source_TM) == len(source_Instance) == len(target_TM) == len(target_img_shapes)
        target_Instances = []
        if source_Instance[0].has('pred_boxes'):
            bbox_key = 'pred_boxes'
        else:
            bbox_key = 'gt_boxes'
        if source_Instance[0].has('pred_boxes'):
            class_key = 'pred_classes'
        else:
            class_key = 'gt_classes' 

        for i,(instance,source_tm,target_tm,target_img_shape) in enumerate(zip(source_Instance,source_TM,target_TM,target_img_shapes)):
            source_boxes = instance.get(bbox_key).tensor
            
            if len(source_boxes)!=0:
                source_points = bbox2points(source_boxes[:, :4])
                source_points = torch.cat(
                            [source_points, source_points.new_ones(source_points.shape[0], 1)], dim=1
                        )
                M_T = np.matmul(target_tm,np.linalg.inv(source_tm))
                M_T = torch.tensor(M_T).to(source_boxes.device).float()
                target_points = (M_T @ source_points.t()).t()
                H,W = target_img_shape
                target_points = target_points[:, :2] / target_points[:, 2:3]
                target_bboxes = points2bbox(target_points, W, H)
                minx,min_y,max_x,max_y = target_bboxes[:,0],target_bboxes[:,1],target_bboxes[:,2],target_bboxes[:,3]
                valid_bboxes = (minx < max_x) & (min_y < max_y)

                target = Instances(target_img_shape)
                target.set(bbox_key,Boxes(target_bboxes[valid_bboxes]))
                if instance.has('scores'):
                    target.scores = instance.scores[valid_bboxes]
                target.set(class_key,instance.get(class_key)[valid_bboxes])
            else:
                target = instance
    
            target_Instances.append(target)

        return target_Instances

    def fcos_losses1(self, gt_classes, gt_shifts_deltas, gt_centerness,
                    pred_class_logits, pred_shift_deltas, pred_centerness,valid_img):
        pred_class_logits, pred_shift_deltas, pred_centerness = \
            permute_all_cls_and_box_to_N_HWA_K_and_concat_fcos(
                pred_class_logits, pred_shift_deltas, pred_centerness,
                self.num_classes
            )
        
        valid_img = (valid_img>0).reshape(gt_classes.shape[0],-1).repeat(1,gt_classes.shape[1]).flatten()
        gt_classes = gt_classes.flatten()
        gt_shifts_deltas = gt_shifts_deltas.view(-1, 4)
        gt_centerness = gt_centerness.view(-1, 1)

        valid_idxs = (gt_classes >= 0) & valid_img
        foreground_idxs = (gt_classes >= 0) & (gt_classes != self.num_classes)
        num_foreground = foreground_idxs.sum()

        gt_classes_target = torch.zeros_like(pred_class_logits)
        gt_classes_target[foreground_idxs, gt_classes[foreground_idxs]] = 1

        loss_cls = sigmoid_focal_loss_jit(
            pred_class_logits[valid_idxs],
            gt_classes_target[valid_idxs],
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        ) / max(1, num_foreground)

        loss_box_reg = iou_loss(
            pred_shift_deltas[foreground_idxs],
            gt_shifts_deltas[foreground_idxs],
            gt_centerness[foreground_idxs],
            box_mode="ltrb",
            loss_type=self.iou_loss_type,
            reduction="mean",
        )

        loss_centerness = F.binary_cross_entropy_with_logits(
            pred_centerness[foreground_idxs],
            gt_centerness[foreground_idxs],
            reduction="sum",
        ) / max(1, num_foreground)

        return {
            "fcos_cls_loss1": loss_cls,
            "fcos_reg_loss1": loss_box_reg,
            "fcos_center_loss1": loss_centerness
        }

    def fcos_losses2(self, gt_classes, gt_shifts_deltas, gt_centerness,
                    pred_class_logits, pred_shift_deltas, pred_centerness,valid_img):
        pred_class_logits, pred_shift_deltas, pred_centerness = \
            permute_all_cls_and_box_to_N_HWA_K_and_concat_fcos(
                pred_class_logits, pred_shift_deltas, pred_centerness,
                self.num_classes
            )

        valid_img = (valid_img>0).reshape(gt_classes.shape[0],-1).repeat(1,gt_classes.shape[1]).flatten()
        gt_classes = gt_classes.flatten()
        gt_shifts_deltas = gt_shifts_deltas.view(-1, 4)
        gt_centerness = gt_centerness.view(-1, 1)

        valid_idxs = (gt_classes >= 0) & valid_img
        foreground_idxs = (gt_classes >= 0) & (gt_classes != self.num_classes)
        num_foreground = foreground_idxs.sum()

        gt_classes_target = torch.zeros_like(pred_class_logits)
        gt_classes_target[foreground_idxs, gt_classes[foreground_idxs]] = 1

        loss_cls = sigmoid_focal_loss_jit(
            pred_class_logits[valid_idxs],
            gt_classes_target[valid_idxs],
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        ) / max(1, num_foreground)

        loss_box_reg = iou_loss(
            pred_shift_deltas[foreground_idxs],
            gt_shifts_deltas[foreground_idxs],
            gt_centerness[foreground_idxs],
            box_mode="ltrb",
            loss_type=self.iou_loss_type,
            reduction="mean",
        )

        loss_centerness = F.binary_cross_entropy_with_logits(
            pred_centerness[foreground_idxs],
            gt_centerness[foreground_idxs],
            reduction="sum",
        ) / max(1, num_foreground)

        return {
            "fcos_cls_loss2": loss_cls,
            "fcos_reg_loss2": loss_box_reg,
            "fcos_center_loss2": loss_centerness
        }

    @torch.no_grad()
    def get_fcos_ground_truth(self, shifts, targets):
        gt_classes = []
        gt_shifts_deltas = []
        gt_centerness = []

        valid_img = torch.ones(len(shifts)).to(shifts[0][0].device)
        for i,(shifts_per_image, targets_per_image) in enumerate(zip(shifts, targets)):

            shifts_over_all_feature_maps = torch.cat(shifts_per_image, dim=0)

            gt_boxes = targets_per_image.gt_boxes

            if len(gt_boxes)==0:
                gt_classes_i = torch.zeros((shifts_over_all_feature_maps.shape[0])) + self.num_classes
                gt_classes_i = gt_classes_i.to(shifts_over_all_feature_maps.device).long()
                gt_shifts_reg_deltas_i = torch.zeros((shifts_over_all_feature_maps.shape[0],4)).to(shifts_over_all_feature_maps.device)
                gt_centerness_i = torch.zeros((shifts_over_all_feature_maps.shape[0])).to(shifts_over_all_feature_maps.device)
                valid_img[i] = 0

            else:
                object_sizes_of_interest = torch.cat([
                    shifts_i.new_tensor(size).unsqueeze(0).expand(
                        shifts_i.size(0), -1) for shifts_i, size in zip(
                        shifts_per_image, self.object_sizes_of_interest)
                ], dim=0)

                deltas = self.shift2box_transform.get_deltas(
                    shifts_over_all_feature_maps, gt_boxes.tensor.unsqueeze(1))

                if self.center_sampling_radius > 0:
                    centers = gt_boxes.get_centers()
                    is_in_boxes = []
                    for stride, shifts_i in zip(self.fpn_strides, shifts_per_image):
                        radius = stride * self.center_sampling_radius
                        center_boxes = torch.cat((
                            torch.max(centers - radius, gt_boxes.tensor[:, :2]),
                            torch.min(centers + radius, gt_boxes.tensor[:, 2:]),
                        ), dim=-1)
                        center_deltas = self.shift2box_transform.get_deltas(
                            shifts_i, center_boxes.unsqueeze(1))
                        is_in_boxes.append(center_deltas.min(dim=-1).values > 0)
                    is_in_boxes = torch.cat(is_in_boxes, dim=1)
                else:
                    is_in_boxes = deltas.min(dim=-1).values > 0

                max_deltas = deltas.max(dim=-1).values
                is_cared_in_the_level = \
                    (max_deltas >= object_sizes_of_interest[None, :, 0]) & \
                    (max_deltas <= object_sizes_of_interest[None, :, 1])

                gt_positions_area = gt_boxes.area().unsqueeze(1).repeat(
                    1, shifts_over_all_feature_maps.size(0))
                gt_positions_area[~is_in_boxes] = math.inf
                gt_positions_area[~is_cared_in_the_level] = math.inf

                positions_min_area, gt_matched_idxs = gt_positions_area.min(dim=0)

                gt_shifts_reg_deltas_i = self.shift2box_transform.get_deltas(
                    shifts_over_all_feature_maps, gt_boxes[gt_matched_idxs].tensor)

                has_gt = len(targets_per_image) > 0
                if has_gt:
                    gt_classes_i = targets_per_image.gt_classes[gt_matched_idxs]
                    gt_classes_i[positions_min_area == math.inf] = self.num_classes
                else:
                    gt_classes_i = torch.zeros_like(
                        gt_matched_idxs) + self.num_classes

                left_right = gt_shifts_reg_deltas_i[:, [0, 2]]
                top_bottom = gt_shifts_reg_deltas_i[:, [1, 3]]
                gt_centerness_i = torch.sqrt(
                    (left_right.min(dim=-1).values / left_right.max(dim=-1).values).clamp_(min=0)
                    * (top_bottom.min(dim=-1).values / top_bottom.max(dim=-1).values).clamp_(min=0)
                )

            gt_classes.append(gt_classes_i)
            gt_shifts_deltas.append(gt_shifts_reg_deltas_i)
            gt_centerness.append(gt_centerness_i)

        return torch.stack(gt_classes), torch.stack(
            gt_shifts_deltas), torch.stack(gt_centerness),valid_img

    def inference(self, box_cls, box_delta, box_center, shifts, images):
        assert len(shifts) == len(images)
        results = []

        box_cls = [permute_to_N_HWA_K(x, self.num_classes) for x in box_cls]
        box_delta = [permute_to_N_HWA_K(x, 4) for x in box_delta]
        box_center = [permute_to_N_HWA_K(x, 1) for x in box_center]

        for img_idx, shifts_per_image in enumerate(shifts):
            image_size = images.image_sizes[img_idx]
            box_cls_per_image = [
                box_cls_per_level[img_idx] for box_cls_per_level in box_cls
            ]
            box_reg_per_image = [
                box_reg_per_level[img_idx] for box_reg_per_level in box_delta
            ]
            box_ctr_per_image = [
                box_ctr_per_level[img_idx] for box_ctr_per_level in box_center
            ]
            results_per_image = self.inference_single_image(
                box_cls_per_image, box_reg_per_image, box_ctr_per_image,
                shifts_per_image, tuple(image_size))
            results.append(results_per_image)
        return results

    def pseudo_gt_generate(self, box_cls, box_delta, box_center, shifts, images,pseudo_score_thres=None):
        assert len(shifts) == len(images)
        results = []
        if pseudo_score_thres:
            thres = pseudo_score_thres
        else:
            thres = self.pseudo_score_thres
        box_cls = [permute_to_N_HWA_K(x, self.num_classes) for x in box_cls]
        box_delta = [permute_to_N_HWA_K(x, 4) for x in box_delta]
        box_center = [permute_to_N_HWA_K(x, 1) for x in box_center]

        for img_idx, shifts_per_image in enumerate(shifts):
            image_size = images.image_sizes[img_idx]
            box_cls_per_image = [
                box_cls_per_level[img_idx] for box_cls_per_level in box_cls
            ]
            box_reg_per_image = [
                box_reg_per_level[img_idx] for box_reg_per_level in box_delta
            ]
            box_ctr_per_image = [
                box_ctr_per_level[img_idx] for box_ctr_per_level in box_center
            ]
            results_per_image = self.pseudo_gt_generate_per_image(
                box_cls_per_image, box_reg_per_image, box_ctr_per_image,
                shifts_per_image, tuple(image_size),thres)
            results.append(results_per_image)
        return results

    def inference_single_image(self, box_cls, box_delta, box_center, shifts,
                                    image_size):
        boxes_all = []
        scores_all = []
        class_idxs_all = []

        for box_cls_i, box_reg_i, box_ctr_i, shifts_i in zip(
                box_cls, box_delta, box_center, shifts):
            box_cls_i = box_cls_i.flatten().sigmoid_()

            num_topk = min(self.topk_candidates, box_reg_i.size(0))
            predicted_prob, topk_idxs = box_cls_i.sort(descending=True)
            predicted_prob = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[:num_topk]

            keep_idxs = predicted_prob > self.score_threshold
            predicted_prob = predicted_prob[keep_idxs]
            topk_idxs = topk_idxs[keep_idxs]

            shift_idxs = topk_idxs // self.num_classes
            classes_idxs = topk_idxs % self.num_classes

            box_reg_i = box_reg_i[shift_idxs]
            shifts_i = shifts_i[shift_idxs]
            predicted_boxes = self.shift2box_transform.apply_deltas(
                box_reg_i, shifts_i)

            box_ctr_i = box_ctr_i.flatten().sigmoid_()[shift_idxs]
            predicted_prob = torch.sqrt(predicted_prob * box_ctr_i)

            boxes_all.append(predicted_boxes)
            scores_all.append(predicted_prob)
            class_idxs_all.append(classes_idxs)

        boxes_all, scores_all, class_idxs_all = [
            cat(x) for x in [boxes_all, scores_all, class_idxs_all]
        ]
        keep = batched_nms(boxes_all, scores_all, class_idxs_all,
                           self.nms_threshold)
        keep = keep[:self.max_detections_per_image]

        result = Instances(image_size)
        result.pred_boxes = Boxes(boxes_all[keep])
        result.scores = scores_all[keep]
        result.pred_classes = class_idxs_all[keep]
        return result

    def pseudo_gt_generate_per_image(self, box_cls, box_delta, box_center, shifts,
                                    image_size,thres):
        boxes_all = []
        scores_all = []
        class_idxs_all = []

        for box_cls_i, box_reg_i, box_ctr_i, shifts_i in zip(
                box_cls, box_delta, box_center, shifts):
            box_cls_i = box_cls_i.flatten().sigmoid_()

            num_topk = min(self.topk_candidates, box_reg_i.size(0))
            predicted_prob, topk_idxs = box_cls_i.sort(descending=True)
            predicted_prob = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[:num_topk]

            keep_idxs = predicted_prob > thres
            predicted_prob = predicted_prob[keep_idxs]
            topk_idxs = topk_idxs[keep_idxs]

            shift_idxs = topk_idxs // self.num_classes
            classes_idxs = topk_idxs % self.num_classes

            box_reg_i = box_reg_i[shift_idxs]
            shifts_i = shifts_i[shift_idxs]
            predicted_boxes = self.shift2box_transform.apply_deltas(
                box_reg_i, shifts_i)

            box_ctr_i = box_ctr_i.flatten().sigmoid_()[shift_idxs]
            predicted_prob = torch.sqrt(predicted_prob * box_ctr_i)

            boxes_all.append(predicted_boxes)
            scores_all.append(predicted_prob)
            class_idxs_all.append(classes_idxs)

        boxes_all, scores_all, class_idxs_all = [
            cat(x) for x in [boxes_all, scores_all, class_idxs_all]
        ]
        keep = batched_nms(boxes_all, scores_all, class_idxs_all,
                           self.nms_threshold)
        keep = keep[:self.max_detections_per_image]

        result = Instances(image_size)
        result.pred_boxes = Boxes(boxes_all[keep])
        result.scores = scores_all[keep]
        result.pred_classes = class_idxs_all[keep]
        return result

    def preprocess_image(self, batched_inputs,teacher_model=None):
        if self.training:
            images_raw = [x["image_Raw"].to(self.device) for x in batched_inputs]
            images_nor = [x["image_Nor"].to(self.device) for x in batched_inputs]
            images_str = [x["image_Str"].to(self.device) for x in batched_inputs]

            images_raw = [self.normalizer(x) for x in images_raw]
            images_nor = [self.normalizer(x) for x in images_nor]
            images_str = [self.normalizer(x) for x in images_str]

            images_raw = ImageList.from_tensors(images_raw,
                                            teacher_model.backbone.size_divisibility)
            images_nor = ImageList.from_tensors(images_nor,
                                            self.backbone.size_divisibility)
            images_str = ImageList.from_tensors(images_str,
                                            self.backbone.size_divisibility)
        else:
            images_nor = [x["image"].to(self.device) for x in batched_inputs]

            images_nor = [self.normalizer(x) for x in images_nor]

            images_nor = ImageList.from_tensors(images_nor,
                                            self.backbone.size_divisibility)
            images_raw = None
            images_str = None

        return images_raw, images_nor,images_str

    def divide_pseudo_gt(self,high_score_Instances,gt_instances_raw,images_raw):
        
        missing_Instances = []
        refine_gt_Instances = []
        for img_inds,(ins,gt_ins) in enumerate(zip(high_score_Instances,gt_instances_raw)):
            image_size = images_raw.image_sizes[img_inds]
            pred_bboxes = ins.pred_boxes.tensor

            if len(pred_bboxes) == 0:
                
                scores = ins.scores
                pred_classes = ins.pred_classes
                gt_bboxes = gt_ins.gt_boxes.tensor.clone()
                gt_classes = gt_ins.gt_classes.clone()
                missing_Instance = Instances(tuple(image_size))
                refine_gt_Instance = Instances(tuple(image_size))
                missing_Instance.pred_boxes = Boxes(pred_bboxes)
                missing_Instance.pred_classes = pred_classes
                missing_Instance.scores = scores
                refine_gt_Instance.gt_boxes = Boxes(gt_bboxes)
                refine_gt_Instance.gt_classes = gt_classes
            else:

                missing_Instance,refine_gt_Instance = Revision_GT(ins,gt_ins)
            
            missing_Instances.append(missing_Instance)
            refine_gt_Instances.append(refine_gt_Instance)
        
        return missing_Instances, refine_gt_Instances


    def teacher_revision(self,teacher_model,features_raw,gt_instances_raw,shifts_raw,images_raw):

        cls_t, delta_t, center_t,_,_,_ = teacher_model.head(features_raw)
        missing_Instances = None
        refine_gt_Instances = None
        high_score_Instances = self.pseudo_gt_generate(cls_t, delta_t, center_t, shifts_raw, images_raw,0.6)
        
        return missing_Instances , refine_gt_Instances,high_score_Instances

class FCOSHead(nn.Module):
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()
        in_channels = input_shape[0].channels
        num_classes = cfg.MODEL.FCOS.NUM_CLASSES
        num_convs = cfg.MODEL.FCOS.NUM_CONVS
        prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.centerness_on_reg = cfg.MODEL.FCOS.CENTERNESS_ON_REG
        self.norm_reg_targets = cfg.MODEL.FCOS.NORM_REG_TARGETS
        cls_subnet = []
        bbox_subnet = []
        for _ in range(num_convs):
            cls_subnet.append(
                nn.Conv2d(in_channels,
                          in_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1))
            cls_subnet.append(nn.GroupNorm(32, in_channels))
            cls_subnet.append(nn.ReLU())
            bbox_subnet.append(
                nn.Conv2d(in_channels,
                          in_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1))
            bbox_subnet.append(nn.GroupNorm(32, in_channels))
            bbox_subnet.append(nn.ReLU())

        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.bbox_subnet = nn.Sequential(*bbox_subnet)
        self.cls_score = nn.Conv2d(in_channels,
                                   num_classes,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
        self.bbox_pred = nn.Conv2d(in_channels,
                                   4,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
        self.centerness = nn.Conv2d(in_channels,
                                    1,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)

        for modules in [
                self.cls_subnet, self.bbox_subnet,
                self.cls_score, self.bbox_pred, self.centerness,
        ]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)
                if isinstance(layer, nn.GroupNorm):
                    torch.nn.init.constant_(layer.weight, 1)
                    torch.nn.init.constant_(layer.bias, 0)

        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_score.bias, bias_value)

        self.scales = nn.ModuleList(
            [Scale(init_value=1.0) for _ in range(len(self.fpn_strides))])

    def forward(self, features, features_color=None):
        logits1 = []
        bbox_reg1 = []
        centerness1 = []
        logits2 = []
        bbox_reg2 = []
        centerness2 = []

        num_scales = [i for i in range(len(self.fpn_strides))]
        if features_color == None:
            features_color = [None for _ in range(len(features))]
        for l, feature, feature_color in zip(num_scales, features, features_color):
            cls_subnet = self.cls_subnet(feature)
            bbox_subnet = self.bbox_subnet(feature)
            logits1.append(self.cls_score(cls_subnet))

            if feature_color is not None:
                cls_subnet_color = self.cls_subnet(feature_color)
                bbox_subnet_color = self.bbox_subnet(feature_color)
                logits2.append(self.cls_score(cls_subnet_color))

            if self.centerness_on_reg:
                centerness1.append(self.centerness(bbox_subnet))
                if feature_color is not None:
                    centerness2.append(self.centerness(bbox_subnet_color))
            else:
                centerness1.append(self.centerness(cls_subnet))
                if feature_color is not None:
                    centerness2.append(self.centerness(cls_subnet_color))

            bbox_pred1 = self.scales[l](self.bbox_pred(bbox_subnet))
            if feature_color is not None:
                bbox_pred2 = self.scales[l](self.bbox_pred(bbox_subnet_color))

            if self.norm_reg_targets:
                bbox_reg1.append(F.relu(bbox_pred1) * self.fpn_strides[l])
                if feature_color is not None:
                    bbox_reg2.append(F.relu(bbox_pred2) * self.fpn_strides[l])
            else:
                bbox_reg1.append(torch.exp(bbox_pred1))
                if feature_color is not None:
                    bbox_reg2.append(torch.exp(bbox_pred2))

        return logits1, bbox_reg1, centerness1, logits2, bbox_reg2, centerness2


        