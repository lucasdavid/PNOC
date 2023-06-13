#!/bin/bash

export PYTHONPATH=$(pwd)

PY=python
SOURCE=scripts/evaluate.py
WORKERS=8

DATASET=voc12
DATA_DIR=/home/ldavid/workspace/datasets/voc/VOCdevkit/VOC2012/
DOMAIN=train
# DOMAIN=val

# DATASET=coco14
# DATA_DIR=$SCRATCH/datasets/coco14/
# DOMAIN=train2014
# DOMAIN=val2014

VERBOSE=1

MIN_TH=0.15
MAX_TH=0.51


run_inference() {
  $PY $SOURCE                   \
    --experiment_name $TAG      \
    --dataset         $DATASET  \
    --domain          $DOMAIN   \
    --data_dir        $DATA_DIR \
    --min_th          $MIN_TH   \
    --max_th          $MAX_TH   \
    --crf_t           $CRF_T       \
    --crf_gt_prob     $CRF_GT_PROB \
    --verbose         $VERBOSE     \
    --num_workers     $WORKERS
}

run_inference_sal() {
  $PY $SOURCE                   \
    --experiment_name $TAG      \
    --dataset         $DATASET  \
    --domain          $DOMAIN   \
    --data_dir        $DATA_DIR \
    --sal_dir         $SAL_DIR  \
    --min_th          $MIN_TH   \
    --max_th          $MAX_TH   \
    --crf_t           $CRF_T       \
    --crf_gt_prob     $CRF_GT_PROB \
    --verbose         $VERBOSE     \
    --num_workers     $WORKERS
}

TAG=occse/rn38d

CRF_T=0
CRF_GT_PROB=0.7
# run_inference

CRF_T=10
CRF_GT_PROB=0.7
run_inference
