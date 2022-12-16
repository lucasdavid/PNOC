#!/bin/bash


export PYTHONPATH=$(pwd)
export OMP_NUM_THREADS=8
export WANDB_PROJECT=research-wsss-dev

PY=python
SOURCE=scripts/cam/train_vanilla.py
DEVICE=cuda
WORKERS=8

DATASET=voc12
DATA_DIR=/home/ldavid/workspace/datasets/voc/VOCdevkit/VOC2012/
# DATASET=coco14
# DATA_DIR=/home/ldavid/workspace/datasets/coco14/

IMAGE_SIZE=384

ARCHITECTURE=resnest50
ARCH=rs50

DILATED=false
MODE=normal
TRAINABLE_STEM=true

AUGMENT=colorjitter_randaugment_cutmix
CUTMIX=0.5
MIXUP=1.
LABELSMOOTHING=0.1

AUG=ra_cm@$CUTMIX
EPOCHS=1
BATCH=16

TAG=$DATASET-$ARCH-$AUG

$PY $SOURCE                          \
  --tag             $TAG             \
  --device          $DEVICE          \
  --num_workers     $WORKERS         \
  --batch_size      $BATCH           \
  --architecture    $ARCHITECTURE    \
  --dilated         $DILATED         \
  --mode            $MODE            \
  --trainable-stem  $TRAINABLE_STEM  \
  --image_size      $IMAGE_SIZE      \
  --min_image_size  $IMAGE_SIZE      \
  --max_image_size  $IMAGE_SIZE      \
  --augment         $AUGMENT         \
  --cutmix_prob     $CUTMIX          \
  --mixup_prob      $MIXUP           \
  --label_smoothing $LABELSMOOTHING  \
  --max_epoch       $EPOCHS          \
  --dataset         $DATASET         \
  --data_dir        $DATA_DIR
