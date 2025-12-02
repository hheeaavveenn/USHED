#!/bin/bash
# U-SHED training script
# Usage: ./train.sh

# Resolve script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# (Optional) Activate conda env
# source /opt/conda/etc/profile.d/conda.sh
# conda activate cv

# Environment variables
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export CVPODS_DATA_ROOT=${CVPODS_DATA_ROOT:-"$PROJECT_ROOT/datasets"}
export OUTPUT_DIR=${OUTPUT_DIR:-"$PROJECT_ROOT/outputs"}
export PRETRAINED_MODEL_PATH=${PRETRAINED_MODEL_PATH:-"$PROJECT_ROOT/pretrained_models/R-50.pkl"}
export PYTHONPATH="$SCRIPT_DIR:$PROJECT_ROOT:${PYTHONPATH:-}"

cd "$PROJECT_ROOT" || exit 1

python train.py --num-gpus 1