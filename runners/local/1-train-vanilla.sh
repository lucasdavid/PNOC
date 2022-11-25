#!/bin/bash

PY=python
SOURCE=train_classification.py
DEVICE=cuda
WORKERS=8
DATA_DIR=/home/ldavid/workspace/datasets/voc/VOCdevkit/VOC2012/

IMAGE_SIZE=64

ARCHITECTURE=resnest101
DILATED=false
MODE=normal
TRAINABLE_STEM=true
AUGMENT=colorjitter_randaugment

TAG=rs101-rr

export PYTHONDONTWRITEBYTECODE=1

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
  --data_dir        $DATA_DIR
