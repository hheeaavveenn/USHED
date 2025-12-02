from __future__ import annotations

import os
from typing import Dict, Any
from datetime import datetime

from .schemas import (
    AugPipelineConfig,
    DynamicPseudoSchedule,
    MPOPlusBlock,
    SFEBlock,
    SmallObjectTuning,
    SolverSchedule,
    TrainerPhases,
    default_weights_path,
)


def _build_model_dict() -> Dict[str, Any]:
    small_obj = SmallObjectTuning().as_dict()
    dynamic_pseudo = DynamicPseudoSchedule().as_dict()
    sfe_block = SFEBlock().as_dict()
    mpo_block = MPOPlusBlock().as_dict()

    return dict(
        DEVICE="cuda",
        WEIGHTS=default_weights_path(),
        RESNETS=dict(
            DEPTH=50,
            FREEZE_AT=1,
        ),
        FCOS=dict(
            NUM_CLASSES=4,
            CENTERNESS_ON_REG=True,
            NORM_REG_TARGETS=True,
            NMS_THRESH_TEST=0.6,
            BBOX_REG_WEIGHTS=(1.0, 1.0, 1.0, 1.0),
            FOCAL_LOSS_GAMMA=2.0,
            FOCAL_LOSS_ALPHA=0.25,
            IOU_LOSS_TYPE="giou",
            CENTER_SAMPLING_RADIUS=1.5,
            OBJECT_SIZES_OF_INTEREST=[
                [-1, 64],
                [64, 128],
                [128, 256],
                [256, 512],
                [512, float("inf")],
            ],
            SMALL_OBJECT_ENHANCEMENT=small_obj,
            PSEUDO_SCORE_THRES=0.5,
            MATCHING_IOU_THRES=0.4,
            WITH_TEACHER=True,
            DYNAMIC_PSEUDO=dynamic_pseudo,
        ),
        SFE=sfe_block,
        MPO_PP=mpo_block,
    )


def _build_dataset_dict(train_set: str) -> Dict[str, Any]:
    return dict(
        CO_MINING=True,
        TRAIN=(train_set,),
        TEST=("duo_val",),
    )


def _build_trainer_dict() -> Dict[str, Any]:
    phases = TrainerPhases().as_dict()
    trainer_dict = dict(
        NAME="MultiBranchRunner",
        DISABLE_AUTO_SCALING=True,
        FP16=dict(
            ENABLED=False,
            TYPE="APEX",
            OPTS=dict(OPT_LEVEL="O1"),
        ),
        WITH_TEACHER=True,
    )
    trainer_dict.update(phases)
    return trainer_dict


def _build_input_block() -> Dict[str, Any]:
    pipelines = AugPipelineConfig()
    return dict(
        AUG=dict(
            TRAIN_PIPELINES=pipelines.build_train_pipelines(),
            TEST_PIPELINES=pipelines.build_test_pipeline(),
        )
    )


def _derive_tag(train_set: str) -> str:
    if train_set.startswith("duo_train_partial_"):
        return train_set.replace("duo_train_partial_", "")
    return train_set


def _build_output_dir(train_set: str) -> str:
    tag = _derive_tag(train_set)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.environ.get("OUTPUT_DIR", "outputs")
    return os.path.join(base, f"duo_DPF_SFE_MPO_{tag}_{stamp}")


def build_ushed_config() -> Dict[str, Any]:
    duo_train_set = "duo_train_partial_50p"
    solver = SolverSchedule().as_dict()
    model_block = _build_model_dict()
    dataset_block = _build_dataset_dict(duo_train_set)
    trainer_block = _build_trainer_dict()
    input_block = _build_input_block()

    output_dir = _build_output_dir(duo_train_set)

    cfg = dict(
        MODEL=model_block,
        DATASETS=dataset_block,
        TRAINER=trainer_block,
        SOLVER=solver,
        TEST=dict(
            EVAL_PERIOD=2000,
            DETECTIONS_PER_IMAGE=100,
        ),
        DATALOADER=dict(
            NUM_WORKERS=8,
            ASPECT_RATIO_GROUPING=True,
            FILTER_EMPTY_ANNOTATIONS=True,
        ),
        INPUT=input_block,
        OUTPUT_DIR=output_dir,
        GLOBAL=dict(
            LOG_INTERVAL=50,
            DUMP_TRAIN=False,
        ),
    )
    return cfg

