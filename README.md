## Enhancing Underwater Object Detection through Hybrid Sparse-Annotation Optimization 
This repository contains the official implementation of **U-SHED**, a hybrid sparse-annotation enhancement framework designed for robust underwater object detection. 
<p align="center">
  <img src="./img.jpg" width="850">
</p>

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

## Pretrained Weights

- U-SHED expects a ResNet-50 backbone checkpoint at:
  - `pretrained_models/R-50.pkl`
- You can use:
  - A COCO-pretrained `R-50` checkpoint compatible with `cvpods`, or
  - Your own ResNet-50 checkpoint trained in `cvpods`.
- Place the file at `pretrained_models/R-50.pkl`, or override the path with:
  - `export PRETRAINED_MODEL_PATH=/absolute/path/to/your/R-50.pkl`

## Quick Start
**Launch training**
   ```bash
   cd /path/to/U-SHED
   python train.py --num-gpus 1
   ```
   
   Or use the provided training script:
   ```bash
   cd /path/to/U-SHED
   ./train.sh
   ```

## Citations
@article{
   title = {Enhancing Underwater Object Detection through Hybrid Sparse-Annotation Optimization},
   url={https://github.com/Y-HeAvEn/USHED},
   journal = {The Visual Computer}   
}
