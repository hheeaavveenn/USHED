from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any


@dataclass(frozen=True)
class DynamicPseudoSchedule:
    enabled: bool = True
    start_iter: int = 0
    end_iter: int = 18000
    score_t_begin: float = 0.50
    score_t_final: float = 0.30
    iou_t_begin: float = 0.25
    iou_t_final: float = 0.45
    per_class: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return dict(
            ENABLED=self.enabled,
            START_ITER=self.start_iter,
            END_ITER=self.end_iter,
            SCORE_T_BEGIN=self.score_t_begin,
            SCORE_T_FINAL=self.score_t_final,
            IOU_T_BEGIN=self.iou_t_begin,
            IOU_T_FINAL=self.iou_t_final,
            PER_CLASS=self.per_class,
        )


@dataclass(frozen=True)
class SFEBlock:
    enabled: bool = True
    weight: float = 0.08
    ema_decay: float = 0.996
    feature_dim: int = 2048
    proj_dim: int = 512
    pred_dim: int = 256
    detach_target: bool = True
    warmup_iters: int = 2000
    use_momentum_schedule: bool = True
    dynamic_weight: bool = False
    feature_alignment_weight: float = 0.02
    feature_alignment_type: str = "cosine"
    progressive_weight_enabled: bool = True
    progressive_weight_max: float = 0.12
    progressive_schedule: str = "cosine"
    progressive_plateau_start: float = 0.2
    progressive_plateau_end: float = 0.7

    def as_dict(self) -> Dict[str, Any]:
        return dict(
            ENABLED=self.enabled,
            WEIGHT=self.weight,
            EMA_DECAY=self.ema_decay,
            FEATURE_DIM=self.feature_dim,
            PROJ_DIM=self.proj_dim,
            PRED_DIM=self.pred_dim,
            DETACH_TARGET=self.detach_target,
            WARMUP_ITERS=self.warmup_iters,
            USE_MOMENTUM_SCHEDULE=self.use_momentum_schedule,
            DYNAMIC_WEIGHT=self.dynamic_weight,
            FEATURE_ALIGNMENT=dict(
                ENABLED=True,
                WEIGHT=self.feature_alignment_weight,
                ALIGNMENT_TYPE=self.feature_alignment_type,
            ),
            PROGRESSIVE_WEIGHT=dict(
                ENABLED=self.progressive_weight_enabled,
                MAX_WEIGHT=self.progressive_weight_max,
                SCHEDULE_TYPE=self.progressive_schedule,
                PLATEAU_START=self.progressive_plateau_start,
                PLATEAU_END=self.progressive_plateau_end,
            ),
        )


@dataclass(frozen=True)
class MPOPlusBlock:
    enabled: bool = True
    topk: int = 300
    small_ratio: float = 0.05
    iou_match_thr: float = 0.45
    scale_k: float = 10.0
    alpha_s: float = 0.4
    alpha_c: float = 0.3
    alpha_sem: float = 0.3
    cosine: bool = True
    safe_relu: bool = True
    quality_thr: float = 0.42

    def as_dict(self) -> Dict[str, Any]:
        return dict(
            ENABLED=self.enabled,
            TOPK_PER_IMAGE=self.topk,
            SMALL_SCALE_RATIO=self.small_ratio,
            IOU_MATCH_THR=self.iou_match_thr,
            SCALE_K=self.scale_k,
            ALPHA_S=self.alpha_s,
            ALPHA_C=self.alpha_c,
            ALPHA_SEM=self.alpha_sem,
            USE_COSINE=self.cosine,
            COSINE_SAFE_RELU=self.safe_relu,
            QUALITY_THR=self.quality_thr,
        )


@dataclass(frozen=True)
class TrainerPhases:
    start_splg_iter: int = 500
    window_size: int = 30
    phase_1_iters: int = 3000
    phase_2_iters: int = 18000
    phase_3_iters: int = 30000
    sfe_start_iter: int = 2000
    progressive_sfe: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return dict(
            START_SPLG_ITER=self.start_splg_iter,
            WINDOW_SIZE=self.window_size,
            SFE_TRAINING=dict(
                PHASE_1_ITERS=self.phase_1_iters,
                PHASE_2_ITERS=self.phase_2_iters,
                PHASE_3_ITERS=self.phase_3_iters,
                SFE_START_ITER=self.sfe_start_iter,
                PROGRESSIVE_SFE=self.progressive_sfe,
            ),
        )


@dataclass(frozen=True)
class SolverSchedule:
    max_iter: int = 30000
    steps: Tuple[int, int] = (18000, 27000)
    warmup_iters: int = 1000
    gamma: float = 0.1
    ims_per_device: int = 4
    ims_per_batch: int = 4
    base_lr: float = 0.0012
    momentum: float = 0.9
    weight_decay: float = 0.0001
    clip_type: str = "value"
    clip_value: float = 0.5

    def as_dict(self) -> Dict[str, Any]:
        return dict(
            CHECKPOINT_PERIOD=2000,
            IMS_PER_DEVICE=self.ims_per_device,
            IMS_PER_BATCH=self.ims_per_batch,
            LR_SCHEDULER=dict(
                MAX_ITER=self.max_iter,
                MAX_EPOCH=None,
                STEPS=self.steps,
                GAMMA=self.gamma,
                WARMUP_ITERS=self.warmup_iters,
                WARMUP_METHOD="linear",
                EPOCH_ITERS=-1,
                EPOCH_WISE=False,
            ),
            OPTIMIZER=dict(
                NAME="SGD",
                BASE_LR=self.base_lr,
                MOMENTUM=self.momentum,
                WEIGHT_DECAY=self.weight_decay,
                CLIP_GRADIENTS=dict(
                    ENABLED=True,
                    CLIP_TYPE=self.clip_type,
                    CLIP_VALUE=self.clip_value,
                ),
            ),
        )


@dataclass(frozen=True)
class SmallObjectTuning:
    enabled: bool = True
    pyramid_weights: Tuple[float, ...] = (0.35, 0.35, 0.20, 0.10)
    loss_weight: float = 1.3
    min_size_train: int = 32

    def as_dict(self) -> Dict[str, Any]:
        return dict(
            ENABLED=self.enabled,
            FEATURE_PYRAMID_WEIGHTS=list(self.pyramid_weights),
            SMALL_OBJECT_LOSS_WEIGHT=self.loss_weight,
            MIN_SIZE_TRAIN=self.min_size_train,
        )


@dataclass(frozen=True)
class AugPipelineConfig:
    underwater_block: Dict[str, Any] = field(
        default_factory=lambda: dict(
            type="UnderwaterEnhancementTransform",
            enable_color_correction=True,
            enable_contrast_enhancement=True,
            enable_dehazing=True,
            enable_illumination_correction=True,
            enhancement_weight=1.0,
            prob=1.0,
        )
    )

    @staticmethod
    def base_resize(short=800, max_size=1333) -> Dict[str, Any]:
        return dict(
            type="RandResizeShortestEdge",
            short_edge_length=(short,),
            max_size=max_size,
            sample_style="choice",
        )

    def build_train_pipelines(self) -> List[List[Dict[str, Any]]]:
        return [
            [
                dict(
                    type="OrderList",
                    transforms=[self.base_resize()],
                    record=True,
                )
            ],
            [
                dict(
                    type="OrderList",
                    transforms=[
                        self.base_resize(),
                        dict(type="RandFlip", prob=0.5),
                        dict(type="ColorJiter"),
                    ],
                    record=True,
                )
            ],
            [
                dict(
                    type="OrderList",
                    transforms=[
                        self.underwater_block,
                        self.base_resize(),
                        dict(type="RandFlip", prob=0.5),
                    ],
                    record=True,
                )
            ],
        ]

    def build_test_pipeline(self) -> List[Any]:
        return [
            (
                "ResizeShortestEdge",
                dict(short_edge_length=800, max_size=1333, sample_style="choice"),
            )
        ]


def default_weights_path() -> str:
    return os.environ.get("PRETRAINED_MODEL_PATH", "pretrained_models/R-50.pkl")

