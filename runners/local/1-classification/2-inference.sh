#!/bin/bash


export PYTHONPATH=$(pwd)
export WANDB_PROJECT=research-wsss-dev

PY=python
SOURCE=scripts/cam/inference.py
DEVICE=cuda
DEVICES=0
WORKERS=8

DATASET=voc12
DATA_DIR=/home/ldavid/workspace/datasets/voc/VOCdevkit/VOC2012/
DOMAIN=train
# DATASET=coco14
# DATA_DIR=/home/ldavid/workspace/datasets/coco14/
# DOMAIN=train2014

ARCHITECTURE=resnest269
DILATED=false
MODE=normal
REG=none

run_inference () {
    CUDA_VISIBLE_DEVICES=$DEVICES           \
    $PY $SOURCE                             \
    --architecture   $ARCHITECTURE          \
    --dilated        $DILATED               \
    --regularization $REG                   \
    --mode           $MODE                  \
    --weights        $WEIGHTS               \
    --tag            $TAG                   \
    --domain         $DOMAIN                \
    --dataset        $DATASET               \
    --data_dir       $DATA_DIR
}

# ARCHITECTURE=resnest269
# WEIGHTS=voc12-rs269-poc-ls0.1@rs269ra-r3
# TAG=poc/$WEIGHTS
# DOMAIN=train_aug
# run_inference
# DOMAIN=val
# run_inference

## RA
#
# ARCHITECTURE=resnest269
# WEIGHTS=cam/resnest269@randaug
# TAG=vanilla/rs269ra
# run_inference


## A-P-OC
#
# ARCHITECTURE=resnest269
# WEIGHTS=poc/voc12-rs269-poc-ls0.1@rs269ra-r3
# TAG=$WEIGHTS
# DOMAIN=train_aug
# run_inference
# DOMAIN=val
# run_inference

DOMAIN=train

WEIGHTS=cam/resnest269@randaug
TAG=$WEIGHTS
run_inference

WEIGHTS=puzzle/ResNeSt269@Puzzle@optimal
TAG=$WEIGHTS
# run_inference

WEIGHTS=puzzle/resnest269@puzzlerep
TAG=$WEIGHTS
# run_inference

WEIGHTS=puzzle/resnest269@puzzlerep2
TAG=$WEIGHTS
# run_inference

WEIGHTS=poc/ResNeSt269@PuzzleOc
TAG=$WEIGHTS
# run_inference

# WEIGHTS=poc/voc12-rs269-poc-ls0.1@rs269ra-r3
# run_inference
# WEIGHTS=pnoc/voc12-rs269-pnoc-ls0.1-ow0.0-1.0-1.0-cams-0.2-octis1-amp@rs269ra-r3
# run_inference
