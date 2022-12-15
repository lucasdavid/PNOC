#!/bin/bash


export PYTHONPATH=$(pwd)

PY=python
SOURCE=scripts/evaluate.py
WORKERS=24

DATASET=voc12
DATA_DIR=/home/ldavid/workspace/datasets/voc/VOCdevkit/VOC2012/
DOMAIN=train
# DATASET=coco14
# DATA_DIR=/home/ldavid/workspace/datasets/coco14/
# DOMAIN=train2014

# TAG=coco14/rs50@train@scale=0.5,1.0,1.5,2.0
TAG=poc/ResNeSt269@PuzzleOc@train@scale=0.5,1.0,1.5,2.0

THRESHOLD=0.3

CUDA_VISIBLE_DEVICES=""          \
    $PY $SOURCE                  \
    --experiment_name $TAG       \
    --num_workers     $WORKERS   \
    --dataset         $DATASET   \
    --domain          $DOMAIN    \
    --data_dir        $DATA_DIR
    # --threshold       $THRESHOLD \
    # --pred_dir        path/to/predictions/
    # --sal_dir         path/to/saliencies/
    # --sal_mode        saliency|segmentation
    # --sal_threshold   0.5
    # --crf_t           10
    # --crf_gt_prob     0.7
