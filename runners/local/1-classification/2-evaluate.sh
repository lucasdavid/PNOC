#!/bin/bash


export PYTHONPATH=$(pwd)
export WANDB_PROJECT=research-wsss-dev

PY=python
SOURCE=scripts/cam/evaluate.py
DEVICE=cuda
WORKERS=8

# DATASET=voc12
# DATA_DIR=/home/ldavid/workspace/datasets/voc/VOCdevkit/VOC2012/
# DOMAIN=val
DATASET=coco14
DATA_DIR=/home/ldavid/workspace/datasets/coco14/
DOMAIN=val2014
BATCH=2

ARCHITECTURE=resnest101
DILATED=false
MODE=normal
REG=none
WEIGHTS=coco14-rs101

TAG=evaluate-$WEIGHTS-$DOMAIN

CUDA_VISIBLE_DEVICES=0                      \
    $PY $SOURCE                             \
    --architecture   $ARCHITECTURE          \
    --device         $DEVICE                \
    --dilated        $DILATED               \
    --regularization $REG                   \
    --mode           $MODE                  \
    --weights        $WEIGHTS               \
    --tag            $TAG                   \
    --domain         $DOMAIN                \
    --batch_size     $BATCH                 \
    --dataset        $DATASET               \
    --max_steps      10000                  \
    --data_dir       $DATA_DIR              &

wait
