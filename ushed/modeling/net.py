import logging

from ushed.modeling.fcos import FCOS
from ushed.modeling import (
    BackboneFactory,
    ShiftGeneratorFactory,
    TeacherFactory,
    build_student_model,
)


LOGGER = logging.getLogger(__name__)


def _wire_factories(cfg):
    backbone_factory = BackboneFactory(cfg)
    shift_factory = ShiftGeneratorFactory(cfg)
    cfg.build_backbone = backbone_factory.build
    cfg.build_shift_generator = shift_factory.build


def build_model(cfg):
    _wire_factories(cfg)

    student = build_student_model(cfg, FCOS)
    teacher = TeacherFactory(cfg, FCOS).build()

    LOGGER.info("Student Model:\n%s", student)
    if teacher is not None:
        LOGGER.info("Teacher Model: Created and frozen")

    return student, teacher
