from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

Tensor = torch.Tensor


def pairwise_iou(b1: Tensor, b2: Tensor, eps: float = 1e-6) -> Tensor:
    if b1.numel() == 0 or b2.numel() == 0:
        dev = (
            b1.device
            if b1.numel()
            else (b2.device if b2.numel() else torch.device("cpu"))
        )
        return torch.zeros((b1.size(0), b2.size(0)), device=dev)
    x11, y11, x12, y12 = b1.unbind(-1)
    x21, y21, x22, y22 = b2.unbind(-1)
    a1 = (x12 - x11).clamp(min=0) * (y12 - y11).clamp(min=0)
    a2 = (x22 - x21).clamp(min=0) * (y22 - y21).clamp(min=0)
    lt_x = torch.max(x11[:, None], x21[None, :])
    lt_y = torch.max(y11[:, None], y21[None, :])
    rb_x = torch.min(x12[:, None], x22[None, :])
    rb_y = torch.min(y12[:, None], y22[None, :])
    inter = (rb_x - lt_x).clamp(min=0) * (rb_y - lt_y).clamp(min=0)
    return inter / (a1[:, None] + a2[None, :] - inter + eps)


def cosine_sim(a: Optional[Tensor], b: Optional[Tensor], eps: float = 1e-8) -> Tensor:
    if a is None or b is None or a.numel() == 0 or b.numel() == 0:
        dev = (
            a.device
            if (a is not None and a.numel())
            else (b.device if (b is not None and b.numel()) else torch.device("cpu"))
        )
        return torch.zeros(
            (a.size(0) if a is not None else 0, b.size(0) if b is not None else 0),
            device=dev,
        )
    a = F.normalize(a, dim=-1, eps=eps)
    b = F.normalize(b, dim=-1, eps=eps)
    return a @ b.t()


def greedy_match_same_class(b1: Tensor, l1: Tensor, b2: Tensor, l2: Tensor, iou_thr: float):
    if b1.numel() == 0 or b2.numel() == 0:
        dev = b1.device if b1.numel() else b2.device
        z = torch.empty(0, dtype=torch.long, device=dev)
        return z, z, torch.empty(0, device=dev)
    iou = pairwise_iou(b1, b2)
    iou *= l1[:, None].eq(l2[None, :]).float()
    idx1 = []
    idx2 = []
    ious = []
    m = iou.clone()
    while True:
        v, k = m.view(-1).max(0)
        if v.item() < iou_thr or v.item() <= 0:
            break
        i, j = divmod(k.item(), m.size(1))
        idx1.append(i)
        idx2.append(j)
        ious.append(v.item())
        m[i, :] = -1
        m[:, j] = -1
    if not idx1:
        dev = b1.device
        z = torch.empty(0, dtype=torch.long, device=dev)
        return z, z, torch.empty(0, device=dev)
    return (
        torch.tensor(idx1, device=b1.device),
        torch.tensor(idx2, device=b1.device),
        torch.tensor(ious, device=b1.device),
    )


@dataclass(frozen=True)
class MPOHyperParams:
    topk_per_image: int = 300
    small_scale_ratio: float = 0.05
    iou_match_thr: float = 0.45
    scale_k: float = 10.0
    alpha_s: float = 0.4
    alpha_c: float = 0.3
    alpha_sem: float = 0.3
    thr_ref: float = 0.5
    use_cosine: bool = True
    safe_relu: bool = True

    @staticmethod
    def from_cfg(cfg: Dict) -> "MPOHyperParams":
        return MPOHyperParams(
            topk_per_image=int(cfg.get("TOPK_PER_IMAGE", 300)),
            small_scale_ratio=float(cfg.get("SMALL_SCALE_RATIO", 0.05)),
            iou_match_thr=float(cfg.get("IOU_MATCH_THR", 0.45)),
            scale_k=float(cfg.get("SCALE_K", 10.0)),
            alpha_s=float(cfg.get("ALPHA_S", 0.4)),
            alpha_c=float(cfg.get("ALPHA_C", 0.3)),
            alpha_sem=float(cfg.get("ALPHA_SEM", 0.3)),
            thr_ref=float(cfg.get("THR_REF", 0.5)),
            use_cosine=bool(cfg.get("USE_COSINE", True)),
            safe_relu=bool(cfg.get("COSINE_SAFE_RELU", True)),
        )


class QualityComposer:
    def __init__(self, params: MPOHyperParams, image_size_hw: Tuple[int, int]):
        self.params = params
        self.image_size_hw = image_size_hw

    def apply_topk(self, boxes: Tensor, scores: Tensor, labels: Tensor, feats: Optional[Tensor]):
        if boxes.size(0) <= self.params.topk_per_image:
            return boxes, scores, labels, feats
        keep = torch.topk(scores, k=self.params.topk_per_image).indices
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]
        if feats is not None and feats.size(0) >= keep.max().item() + 1:
            feats = feats[keep]
        return boxes, scores, labels, feats

    def scale_weight(self, boxes: Tensor) -> Tensor:
        if boxes.numel() == 0:
            return boxes.new_zeros((0,))
        h, w = self.image_size_hw
        wh = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0)
        area = (wh[:, 0] * wh[:, 1]).clamp(min=0)
        max_side = float(max(h, w) + 1e-6)
        scale = torch.sqrt(area) / max_side
        ref = self.params.small_scale_ratio
        k = self.params.scale_k
        weights = torch.sigmoid(k * (ref - scale))
        return weights.clamp(0.0, 1.0)

    def consistency_weight(
        self,
        boxes: Tensor,
        labels: Tensor,
        feats_v1: Optional[Tensor],
        preds_v2: Optional[Dict[str, Tensor]],
        feats_v2: Optional[Tensor],
    ) -> Tensor:
        if preds_v2 is None or feats_v1 is None or feats_v2 is None:
            return boxes.new_zeros((boxes.size(0),))
        idx1, idx2, _ = greedy_match_same_class(
            boxes,
            labels,
            preds_v2["boxes"],
            preds_v2.get("labels", None),
            self.params.iou_match_thr,
        )
        scores = boxes.new_zeros((boxes.size(0),))
        if idx1.numel() == 0:
            return scores
        cos = cosine_sim(feats_v1, feats_v2)[idx1, idx2]
        if self.params.safe_relu:
            cos = torch.clamp(cos, min=0.0)
        scores[idx1] = cos
        return scores.clamp(0.0, 1.0)

    def semantic_weight(self, feats: Optional[Tensor], labels: Tensor, scores: Tensor) -> Tensor:
        if feats is None or feats.numel() == 0:
            return scores.new_zeros(scores.shape)
        high = scores >= self.params.thr_ref
        if not high.any():
            return scores.new_zeros(scores.shape)
        labels_high = labels[high]
        feats_high = feats[high]
        sem = scores.new_zeros(scores.shape)
        for class_id in labels_high.unique():
            mask_high = labels_high == class_id
            if not mask_high.any():
                continue
            proto = feats_high[mask_high].mean(dim=0, keepdim=True)
            idx_all = labels == class_id
            if not idx_all.any():
                continue
            feats_c = feats[idx_all]
            sim = F.cosine_similarity(
                F.normalize(feats_c, dim=-1),
                F.normalize(proto.expand_as(feats_c), dim=-1),
                dim=-1,
            )
            sem[idx_all] = torch.clamp(sim, min=0.0, max=1.0)
        return sem

    def compose(
        self,
        preds_v1: Dict[str, Tensor],
        preds_v2: Optional[Dict[str, Tensor]],
        feats_v1: Optional[Tensor],
        feats_v2: Optional[Tensor],
    ) -> Dict[str, Tensor]:
        boxes = preds_v1["boxes"]
        scores = preds_v1["scores"].flatten()
        labels = preds_v1["labels"]
        boxes, scores, labels, feats_v1 = self.apply_topk(boxes, scores, labels, feats_v1)
        w_s = self.scale_weight(boxes)
        w_c = self.consistency_weight(boxes, labels, feats_v1, preds_v2, feats_v2)
        w_sem = self.semantic_weight(feats_v1, labels, scores)

        alpha_s = self.params.alpha_s
        alpha_c = self.params.alpha_c
        alpha_sem = self.params.alpha_sem
        alpha_sum = max(alpha_s + alpha_c + alpha_sem, 1e-6)
        alpha_s /= alpha_sum
        alpha_c /= alpha_sum
        alpha_sem /= alpha_sum

        quality = alpha_s * w_s + alpha_c * w_c + alpha_sem * w_sem
        quality = quality.clamp(0.0, 1.0)

        return dict(
            boxes=boxes,
            scores=scores,
            labels=labels,
            weight=quality,
            score_cal=quality,
            cons=w_c,
            w_s=w_s,
            quality=quality,
            small_mask=(w_s > 0.5),
        )


@torch.no_grad()
def mpopp_weights(
    preds_v1: Dict[str, Tensor],
    preds_v2: Optional[Dict[str, Tensor]],
    feats_v1: Optional[Tensor],
    feats_v2: Optional[Tensor],
    image_size_hw: Tuple[int, int],
    cfg: Dict,
):
    params = MPOHyperParams.from_cfg(cfg)
    composer = QualityComposer(params, image_size_hw)
    if "labels" not in preds_v1:
        raise ValueError("preds_v1 must contain 'labels'")
    return composer.compose(preds_v1, preds_v2, feats_v1, feats_v2)
