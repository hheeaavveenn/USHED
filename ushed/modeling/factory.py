from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

from cvpods.layers import ShapeSpec
from cvpods.modeling.backbone import Backbone
from cvpods.modeling.backbone.fpn import build_retinanet_resnet_fpn_backbone
from cvpods.modeling.anchor_generator import ShiftGenerator

LOGGER = logging.getLogger(__name__)


class BackboneFactory:
    def __init__(self, cfg):
        self.cfg = cfg

    def build(self, input_shape: Optional[ShapeSpec] = None) -> Backbone:
        if input_shape is None:
            input_shape = ShapeSpec(channels=len(self.cfg.MODEL.PIXEL_MEAN))
        backbone = build_retinanet_resnet_fpn_backbone(self.cfg, input_shape)
        if not isinstance(backbone, Backbone):
            raise TypeError("Backbone must inherit from cvpods.modeling.Backbone")
        return backbone


class ShiftGeneratorFactory:
    def __init__(self, cfg):
        self.cfg = cfg

    def build(self, feature_shapes):
        return ShiftGenerator(self.cfg, feature_shapes)


@dataclass
class SFEInjector:
    cfg: object

    def _prepare_config(self) -> dict:
        sfe = self.cfg.MODEL.get("SFE", {})
        payload = dict(
            sfe_weight=sfe.get("WEIGHT", 0.3),
            ema_decay=sfe.get("EMA_DECAY", 0.996),
            proj_dim=sfe.get("PROJ_DIM", 256),
            detach_target=sfe.get("DETACH_TARGET", True),
        )
        payload.pop("feature_dim", None)
        return payload

    def inject(self, model):
        if not self.cfg.MODEL.get("SFE", {}).get("ENABLED", False):
            return model

        from enhancements.ssl.simplified_sfe import create_sfe_enhanced_model

        config_payload = self._prepare_config()
        LOGGER.info("Attaching SFE wrapper with config: %s", config_payload)
        wrapped = create_sfe_enhanced_model(model, config_payload)
        try:
            inner = getattr(wrapped, "ushed", None)
            projector = getattr(getattr(wrapped, "sfe_head", None), "projector", None)
            if inner is not None and projector is not None:
                setattr(inner, "sfe_projector", projector)
                setattr(inner, "sfe_proj_dim", int(config_payload.get("proj_dim", 256)))
                LOGGER.info("Injected projector into inner FCOS instance")
            else:
                LOGGER.warning("Unable to inject projector into inner FCOS instance")
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.warning("SFE projector injection failed: %s", exc)
        return wrapped


class TeacherFactory:
    def __init__(self, cfg, model_cls):
        self.cfg = cfg
        self.model_cls = model_cls

    def build(self):
        if not self.cfg.MODEL.FCOS.WITH_TEACHER:
            return None
        teacher = self.model_cls(self.cfg)
        teacher.freeze_all()
        LOGGER.info("Teacher model created and frozen")
        return teacher


def build_student_model(cfg, model_cls):
    model = model_cls(cfg)
    injector = SFEInjector(cfg)
    model = injector.inject(model)
    return model

