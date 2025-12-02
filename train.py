#!/usr/bin/env python3
"""
U-SHED training entry point.
This script imports the actual config and build_model from ushed.modeling,
then calls cvpods training functions.
"""
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import actual implementations
from ushed.modeling.config import config
from ushed.modeling.net import build_model

# Import cvpods training utilities
from cvpods.engine import RUNNERS, default_argument_parser, default_setup, hooks, launch
from cvpods.evaluation import build_evaluator
from cvpods.modeling import GeneralizedRCNNWithTTA
from cvpods.utils import comm
from collections import OrderedDict
from loguru import logger


def runner_decrator(cls):
    """
    Decorator for runner class to add custom methods.
    Based on cvpods/tools/train_net.py
    """
    def custom_build_evaluator(cls, cfg, dataset_name, dataset, output_folder=None):
        """Create evaluator(s) for a given dataset."""
        dump_train = cfg.GLOBAL.DUMP_TRAIN
        return build_evaluator(cfg, dataset_name, dataset, output_folder, dump=dump_train)

    def custom_test_with_TTA(cls, cfg, model):
        """Run evaluation with test-time augmentation."""
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        res = cls.test(cfg, model, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA"))
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

    cls.build_evaluator = classmethod(custom_build_evaluator)
    cls.test_with_TTA = classmethod(custom_test_with_TTA)

    return cls


@logger.catch
def main(args, config, build_model):
    config.merge_from_list(args.opts)
    cfg = default_setup(config, args)

    runner = runner_decrator(RUNNERS.get(cfg.TRAINER.NAME))(cfg, build_model)
    runner.resume_or_load(resume=args.resume)

    extra_hooks = []
    if args.clearml:
        from cvpods.engine.clearml import ClearMLHook
        if comm.is_main_process():
            extra_hooks.append(ClearMLHook())
    if cfg.TEST.AUG.ENABLED:
        extra_hooks.append(
            hooks.EvalHook(0, lambda: runner.test_with_TTA(cfg, runner.model))
        )
    if extra_hooks:
        runner.register_hooks(extra_hooks)

    logger.info("Running with full config:\n{}".format(cfg))
    base_config = cfg.__class__.__base__()
    logger.info("different config with base class:\n{}".format(cfg.diff(base_config)))

    runner.train()


def train_argument_parser():
    parser = default_argument_parser()
    parser.add_argument("--clearml", action="store_true", help="use clearml or not")
    return parser


if __name__ == "__main__":
    args = train_argument_parser().parse_args()
    if args.num_gpus is None:
        import torch
        args.num_gpus = torch.cuda.device_count()

    config.link_log()
    logger.info("Create soft link to {}".format(config.OUTPUT_DIR))

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args, config, build_model),
    )

