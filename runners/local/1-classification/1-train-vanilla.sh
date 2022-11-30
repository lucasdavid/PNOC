#!/bin/bash

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$(pwd)

PY=python
SOURCE=scripts/cam/train_vanilla.py
DEVICE=cuda
WORKERS=8

# DATASET=voc12
# DATA_DIR=/home/ldavid/workspace/datasets/voc/VOCdevkit/VOC2012/
DATASET=coco14
DATA_DIR=/home/ldavid/workspace/datasets/coco14/

IMAGE_SIZE=64

ARCHITECTURE=resnest101
DILATED=false
MODE=normal
TRAINABLE_STEM=true
AUGMENT=colorjitter_randaugment

TAG=rs101-rr-$DATASET

$PY $SOURCE                          \
  --tag             $TAG             \
  --device          $DEVICE          \
  --num_workers     $WORKERS         \
  --architecture    $ARCHITECTURE    \
  --dilated         $DILATED         \
  --mode            $MODE            \
  --trainable-stem  $TRAINABLE_STEM  \
  --image_size      $IMAGE_SIZE      \
  --augment         $AUGMENT         \
  --dataset         $DATASET         \
  --data_dir        $DATA_DIR
