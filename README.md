This repository contains the official implementation of the manuscript:

**"Enhancing Underwater Object Detection through Hybrid Sparse-Annotation Optimization "**  
submitted to *The Visual Computer* 
.
## Overview

U-SHED (Underwater Sparse-annotation Hybrid Enhancement for Detection) targets sparsely-annotated underwater object detection by combining a three-branch teacher–student architecture with three core modules: Dynamic Pseudo-label Filtering (DPF), Self-supervised Feature Enhancement (SFE, BYOL-style), and Multi-dimensional Pseudo-label Optimization (MPO). The project is built on cvpods and can be run directly from the project root directory.

## Highlights

- **DPF**: Score/IoU dual thresholds are linearly scheduled over training iterations to balance pseudo-label coverage and quality.
- **SFE**: An EMA target branch based on USHEDBYOL stabilizes semantics under strong/underwater augmentations.
- **MPO**: Re-weights pseudo labels by scale, cross-view consistency, and semantic similarity to suppress noisy labels.


## Layout

```
U-SHED/
├── train.py                   # Training entry point
├── train.sh                   # Training script
├── ushed/
│   ├── data/
│   │   └── duo.py             # DUO dataset loader + multi-branch sampler
│   ├── modeling/
│   │   ├── config.py           # Unified configuration (SFE / DPF / MPO)
│   │   ├── fcos.py            # FCOS backbone + DPF scheduling + MPO entry
│   │   ├── net.py             # Build Student/Teacher and inject SFE
│   │   ├── mpo.py             # MPO weight computation (scale/consistency/semantic)
│   │   ├── dpf.py             # DPF scheduler
│   │   └── factory.py         # Model factory components
│   └── config/
│       ├── schemas.py         # Configuration schemas
│       └── builder.py         # Configuration builder
├── enhancements/
│   ├── ssl/simplified_byol.py
│   ├── ssl/simplified_sfe.py
│   └── underwater/underwater_enhancement.py
├── datasets/DUO/              # (Optional) example dataset layout
├── pretrained_models/R-50.pkl # (Optional) example pretrained weights
├── requirements.txt            # Python dependencies
└── README.md
```

> To use your own data and weights, simply replace `datasets/` and `pretrained_models/` and set the environment variables.

## Environment

- Python ≥ 3.8, CUDA ≥ 11.0
- PyTorch ≥ 1.8
- cvpods (`git clone https://github.com/Megvii-BaseDetection/cvpods.git && pip install -e .`)
- Dependencies: `torchvision opencv-python numpy pycocotools matplotlib pillow`

```bash
pip install -r requirements.txt
```

## Quick Start

1. **Set environment variables**
   ```bash
   export CVPODS_DATA_ROOT=/path/to/datasets
   export PRETRAINED_MODEL_PATH=/path/to/pretrained_models/R-50.pkl
   export OUTPUT_DIR=/path/to/outputs
   export CUDA_VISIBLE_DEVICES=0
   ```
2. **Launch training**
   ```bash
   cd /path/to/U-SHED
   python train.py --num-gpus 1
   ```
   
   Or use the provided training script:
   ```bash
   cd /path/to/U-SHED
   ./train.sh
   ```

## Pretrained Weights

- U-SHED expects a ResNet-50 backbone checkpoint at:
  - `pretrained_models/R-50.pkl`
- You can use:
  - A COCO-pretrained `R-50` checkpoint compatible with `cvpods`, or
  - Your own ResNet-50 checkpoint trained in `cvpods`.
- Place the file at `pretrained_models/R-50.pkl`, or override the path with:
  - `export PRETRAINED_MODEL_PATH=/absolute/path/to/your/R-50.pkl`

## Dataset Layout

```
datasets/DUO/
├── images/{train,test}/
└── annotations/
    ├── instances_train.json
    ├── train_sparse_10p.json
    ├── train_sparse_30p.json
    ├── train_sparse_50p.json
    └── instances_test.json
```

## Module Notes

- **DPF**: Implemented in `ushed/modeling/fcos.py` via `_dpf_dynamic_score_thr`, `_dpf_dynamic_iou_thr`, and `pseudo_gt_generate` for dynamic threshold scheduling and pseudo-label filtering. The scheduler logic is encapsulated in `ushed/modeling/dpf.py`.
- **SFE**: `enhancements/ssl/simplified_byol.py` (USHEDBYOL) and `simplified_sfe.py` provide the BYOL head; `ushed/modeling/net.py` injects SFE according to `MODEL.SFE`.
- **MPO**: `ushed/modeling/mpo.py` produces quality weights from scale (`w_s`), cross-view consistency (`w_c`), and semantic similarity (`w_sem`), and `ushed/modeling/fcos.py` applies them to pseudo labels.

