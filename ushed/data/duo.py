from __future__ import annotations

import copy
import logging
import os
import os.path as osp
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from cvpods.data.datasets.coco import COCODataset
from cvpods.data.detection_utils import (
    annotations_to_instances,
    check_image_size,
    filter_empty_instances,
    read_image,
)
from cvpods.data.registry import DATASETS, PATH_ROUTES


LOGGER = logging.getLogger(__name__)


DUO_SPLITS: Dict[str, Tuple[str, str]] = {
    "duo_train": ("DUO/images/train", "DUO/annotations/instances_train.json"),
    "duo_train_partial_30p": (
        "DUO/images/train",
        "DUO/annotations/train_sparse_30p.json",
    ),
    "duo_train_partial_50p": (
        "DUO/images/train",
        "DUO/annotations/train_sparse_50p.json",
    ),
    "duo_train_partial_70p": (
        "DUO/images/train",
        "DUO/annotations/train_sparse_70p.json",
    ),
    "duo_train_partial_10p": (
        "DUO/images/train",
        "DUO/annotations/train_sparse_10p.json",
    ),
    "duo_val": ("DUO/images/test", "DUO/annotations/instances_test.json"),
}


def _register_duo_splits() -> None:
    """
    Ensure DUO dataset splits are visible to cvpods through PATH_ROUTES.
    """
    if "DUO" not in PATH_ROUTES._obj_map:
        routes: Dict[str, Any] = {
            "dataset_type": "DUOMultiBranch",
            "evaluator_type": {"duo": "coco"},
            "duo": dict(DUO_SPLITS),
        }
        PATH_ROUTES.register(routes, "DUO")
        return

    routes = PATH_ROUTES.get("DUO")
    routes["dataset_type"] = "DUOMultiBranch"
    duo_cfg = routes.get("duo", {})
    duo_cfg.update(DUO_SPLITS)
    routes["duo"] = duo_cfg


_register_duo_splits()

BRANCH_TAGS: Tuple[str, str, str] = ("Raw", "Nor", "Str")


def _default_data_root() -> str:
    root = osp.join(
        osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))),
        "datasets",
    )
    return root if osp.isdir(root) else ""


def _filter_missing_files(dataset_dicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    missing = 0
    for dd in dataset_dicts:
        file_path = dd.get("file_name")
        if file_path and osp.exists(file_path):
            kept.append(dd)
        else:
            missing += 1
    if missing > 0:
        LOGGER.warning(
            "[DUO] filtered out %d missing images; keep %d",
            missing,
            len(kept),
        )
    return kept


def _resolve_existing_sample(
    index: int, dataset_dicts: List[Dict[str, Any]]
) -> Tuple[int, Dict[str, Any]]:
    """
    Resolve an index to a dataset dict whose image file actually exists.
    """
    candidate = copy.deepcopy(dataset_dicts[index])
    path = candidate.get("file_name")
    if path and osp.exists(path):
        return index, candidate

    LOGGER.warning("Image file not found, searching for valid sample: %s", path)

    n = len(dataset_dicts)
    for offset in range(1, min(100, n)):
        nxt = (index + offset) % n
        candidate = copy.deepcopy(dataset_dicts[nxt])
        path = candidate.get("file_name")
        if path and osp.exists(path):
            return nxt, candidate

    for nxt in range(n):
        candidate = copy.deepcopy(dataset_dicts[nxt])
        path = candidate.get("file_name")
        if path and osp.exists(path):
            return nxt, candidate

    return index, copy.deepcopy(dataset_dicts[index])


@DATASETS.register()
class DUOMultiBranch(COCODataset):
    def __init__(self, config, dataset_name, transforms=None, is_train=True):
        super().__init__(config, dataset_name, transforms, is_train)
        self.task_key = "duo"
        self.data_root = os.environ.get("CVPODS_DATA_ROOT", _default_data_root())

        if hasattr(self, "dataset_dicts") and isinstance(self.dataset_dicts, list):
            self.dataset_dicts = _filter_missing_files(self.dataset_dicts)

    def _get_metadata(self, dataset_name: str) -> Dict[str, Any]:
        from cvpods.data.datasets.builtin_meta import _get_builtin_metadata

        meta = _get_builtin_metadata("coco")
        duo_routes = PATH_ROUTES.get("DUO")

        if dataset_name not in duo_routes["duo"]:
            raise KeyError(f"Dataset {dataset_name} not found in DUO routes")

        image_root, json_file = duo_routes["duo"][dataset_name]
        meta["image_root"] = (
            osp.join(self.data_root, image_root)
            if "://" not in image_root
            else image_root
        )
        meta["json_file"] = (
            osp.join(self.data_root, json_file) if "://" not in json_file else json_file
        )
        meta["evaluator_type"] = duo_routes["evaluator_type"].get("duo", "coco")
        meta["thing_classes"] = ["holothurian", "echinus", "scallop", "starfish"]
        meta["thing_dataset_id_to_contiguous_id"] = {1: 0, 2: 1, 3: 2, 4: 3}
        return meta

    def _apply_transforms(self, image, annotations=None, **kwargs):
        if self.transforms is None:
            return image, annotations

        if isinstance(self.transforms, dict):
            result: Dict[str, Tuple[Any, Any]] = {}
            for key, tfms in self.transforms.items():
                img = copy.deepcopy(image)
                ann = copy.deepcopy(annotations)
                for tfm in tfms:
                    img, ann = tfm(img, ann, **kwargs)
                result[key] = (img, ann)
            return result, None

        if isinstance(self.transforms, list):
            tfm = self.transforms[kwargs["index"]]
            return tfm(image, annotations, **kwargs)

        for tfm in self.transforms:
            image, annotations = tfm(image, annotations, **kwargs)
        return image, annotations

    def __getitem__(self, index: int) -> Dict[str, Any]:
        index, dataset_dict = _resolve_existing_sample(index, self.dataset_dicts)
        image_path = dataset_dict["file_name"]

        image = read_image(image_path, format=self.data_format)
        check_image_size(dataset_dict, image)

        if "annotations" in dataset_dict:
            annotations = dataset_dict.pop("annotations")
            annotations = [ann for ann in annotations if ann.get("iscrowd", 0) == 0]
        else:
            annotations = None

        annotations_copy = copy.deepcopy(annotations)
        image_orig = torch.as_tensor(image.copy(), dtype=torch.uint8)
        dataset_dict["image_orig"] = image_orig

        if len(self.transforms) == 1:
            image_i, annotations_i = self._apply_transforms(
                image, annotations, index=0
            )
            image_shape = image_i.shape[:2]
            instances = annotations_to_instances(
                annotations_i, image_shape, mask_format=self.mask_format
            )
            dataset_dict["instances"] = filter_empty_instances(instances)
            dataset_dict["image"] = torch.as_tensor(
                np.ascontiguousarray(image_i.transpose(2, 0, 1))
            )
            return dataset_dict

        for branch_idx, branch_tag in enumerate(BRANCH_TAGS):
            ann = copy.deepcopy(annotations_copy)
            image_i, annotations_i = self._apply_transforms(
                image, ann, index=branch_idx
            )
            image_shape = image_i.shape[:2]
            instances = annotations_to_instances(
                annotations_i, image_shape, mask_format=self.mask_format
            )
            dataset_dict[f"instances_{branch_tag}"] = filter_empty_instances(
                instances
            )

            if annotations_i and "transform_matrix" in annotations_i[0]:
                dataset_dict[f"transform_matrix_{branch_tag}"] = annotations_i[0][
                    "transform_matrix"
                ]

            dataset_dict[f"image_{branch_tag}"] = torch.as_tensor(
                np.ascontiguousarray(image_i.transpose(2, 0, 1))
            )

        return dataset_dict
