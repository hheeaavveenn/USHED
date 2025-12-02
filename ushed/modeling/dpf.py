from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DPFScheduler:
    cfg: dict
    default_score_thr: float
    default_iou_thr: float
    max_iter: int

    def _enabled(self) -> bool:
        return bool(self.cfg) and bool(self.cfg.get("ENABLED", False)) and self.max_iter > 0

    def score_thr(self, curr_iter: int) -> float:
        if not self._enabled():
            return float(self.default_score_thr)
        dp = self.cfg
        s0 = float(dp.get("SCORE_T_BEGIN", 0.8))
        s1 = float(dp.get("SCORE_T_FINAL", 0.4))
        it0 = int(dp.get("START_ITER", 0))
        it1 = int(dp.get("END_ITER", self.max_iter))
        it = max(it0, min(curr_iter, it1))
        if it1 <= it0:
            return float(s1)
        ratio = (it - it0) / float(it1 - it0)
        thr = s0 + (s1 - s0) * ratio
        return float(thr)

    def iou_thr(self, curr_iter: int) -> float:
        if not self._enabled():
            return float(self.default_iou_thr)
        dp = self.cfg
        iou0 = float(dp.get("IOU_T_BEGIN", 0.25))
        iou1 = float(dp.get("IOU_T_FINAL", 0.45))
        it0 = int(dp.get("START_ITER", 0))
        it1 = int(dp.get("END_ITER", self.max_iter))
        it = max(it0, min(curr_iter, it1))
        if it1 <= it0:
            return float(iou1)
        ratio = (it - it0) / float(it1 - it0)
        thr = iou0 + (iou1 - iou0) * ratio
        return float(thr)


