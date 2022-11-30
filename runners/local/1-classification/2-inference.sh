#!/bin/bash

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$(pwd)

PY=python
SOURCE=scripts/cam/inference.py
DEVICE=cuda
WORKERS=8

# DATASET=voc12
# DATA_DIR=/home/ldavid/workspace/datasets/voc/VOCdevkit/VOC2012/
DATASET=coco14
DATA_DIR=/home/ldavid/workspace/datasets/coco14/

DOMAIN=train2014

ARCHITECTURE=resnest50
DILATED=false
MODE=normal
REG=none
WEIGHTS=coco14-rs50

TAG=$WEIGHTS

CUDA_VISIBLE_DEVICES=0                      \
    $PY $SOURCE                             \
    --architecture   $ARCHITECTURE          \
    --dilated        $DILATED               \
    --regularization $REG                   \
    --mode           $MODE                  \
    --weights        $WEIGHTS               \
    --tag            $TAG                   \
    --domain         $DOMAIN                \
    --dataset        $DATASET               \
    --data_dir       $DATA_DIR              &

wait
