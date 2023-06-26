#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH -p sequana_gpu_shared
#SBATCH -J tr-aff
#SBATCH -o /scratch/lerdl/lucas.david/logs/puzzle/affinitynet/train-%j.out
#SBATCH --time=24:00:00

### 48:00:00

# Copyright 2021 Lucas Oliveira David
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#
# Random Walk.
#

# export OMP_NUM_THREADS=4

# Environment
### Sdumont
# PY=python3.8
# DEVICES=0,1,2,3
# WORKERS=8
# WORK_DIR=$SCRATCH/PuzzleCAM
# DATA_DIR=$SCRATCH/datasets

## Local
PY=python
DEVICES=0
WORKERS=8
WORK_DIR=/home/ldavid/workspace/repos/research/pnoc
DATA_DIR=/home/ldavid/workspace/datasets

cd $WORK_DIR
export PYTHONPATH=$(pwd)

# Dataset
## Pascal VOC 2012
DATASET=voc12
DOMAIN=train_aug
DATA_DIR=$DATA_DIR/voc/VOCdevkit/VOC2012/

IMAGE_SIZE=512
MIN_IMAGE_SIZE=320
MAX_IMAGE_SIZE=640

### MS COCO 2014
# DATASET=coco14
# DOMAIN=train2014
# DATA_DIR=$DATA_DIR/coco14/

# IMAGE_SIZE=640
# MIN_IMAGE_SIZE=400
# MAX_IMAGE_SIZE=800

# Architecture
ARCHITECTURE=resnest269
ARCH=rs269
BATCH_SIZE=32
LR=0.1

# Infrastructure
MIXED_PRECISION=true # false


rw_make_affinity_labels() {
  echo "=================================================================="
  echo "[rw make affinity labels] started at $(date +'%Y-%m-%d %H:%M:%S')."
  echo "=================================================================="

  $PY scripts/rw/rw_make_affinity_labels.py \
    --tag $AFF_LABELS_TAG \
    --dataset $DATASET \
    --domain $DOMAIN \
    --fg_threshold $FG \
    --bg_threshold $BG \
    --crf_t $CRF_T \
    --crf_gt_prob $CRF_GT \
    --cams_dir $CAMS_DIR \
    --sal_dir $SAL_DIR \
    --data_dir $DATA_DIR \
    --num_workers $WORKERS
}

rw_training() {
  echo "=================================================================="
  echo "[rw training] started at $(date +'%Y-%m-%d %H:%M:%S')."
  echo "=================================================================="

  CUDA_VISIBLE_DEVICES=$DEVICES \
    WANDB_TAGS="$DATASET,$ARCH,rw" \
    $PY scripts/rw/train_affinity.py \
    --architecture $ARCHITECTURE \
    --tag $AFF_TAG \
    --batch_size $BATCH_SIZE \
    --image_size $IMAGE_SIZE \
    --min_image_size $MIN_IMAGE_SIZE \
    --max_image_size $MAX_IMAGE_SIZE \
    --dataset $DATASET \
    --lr $LR \
    --label_dir $AFF_LABELS_DIR \
    --data_dir $DATA_DIR \
    --num_workers $WORKERS
}

rw_inference() {
  echo "=================================================================="
  echo "[rw inference $DOMAIN] started at $(date +'%Y-%m-%d %H:%M:%S')."
  echo "=================================================================="

  CUDA_VISIBLE_DEVICES=$DEVICES \
    $PY scripts/rw/inference.py \
    --architecture $ARCHITECTURE \
    --image_size $IMAGE_SIZE \
    --model_name $AFF_TAG \
    --cam_dir $CAMS_DIR \
    --domain $DOMAIN \
    --beta $RW_BETA \
    --exp_times $RW_EXP \
    --mixed_precision $MIXED_PRECISION \
    --dataset $DATASET \
    --data_dir $DATA_DIR
}

make_pseudo_labels() {
  echo "=================================================================="
  echo "[rw make pseudo labels] started at $(date +'%Y-%m-%d %H:%M:%S')."
  echo "=================================================================="

  $PY scripts/segmentation/make_pseudo_labels.py \
    --experiment_name $AFF_TAG \
    --dataset $DATASET \
    --domain $DOMAIN \
    --threshold $THRESHOLD \
    --crf_t $CRF_T \
    --crf_gt_prob $CRF_GT \
    --data_dir $DATA_DIR
}


## 3.1 Make Affinity Labels
##
# PRIORS_TAG=ra-oc-p-poc-pnoc-avg
PRIORS_TAG=ra-oc-p-poc-pnoc-learned-a0.25

CAMS_DIR=./experiments/predictions/ensemble/$PRIORS_TAG
SAL_DIR=./experiments/predictions/saliency/voc12-pn@ccamh-rs269@$PRIORS_TAG
FG=0.3
BG=0.1
CRF_T=10
CRF_GT=0.7

AFF_LABELS_TAG=rw/$DATASET-an@ccamh@$PRIORS_TAG@crf$CRF_T-gt$CRF_GT
rw_make_affinity_labels

## 3.2. Affinity Net Train
##
AFF_TAG=rw/$DATASET-an@ccamh@$PRIORS_TAG
AFF_LABELS_DIR=./experiments/predictions/$AFF_LABELS_TAG@aff_fg="$FG"_bg="$BG"
rw_training

## 3.3. Affinity Net Inference
##
RW_DIR=$AN_WEIGHTS@$DOMAIN@beta=$RW_BETA@exp_times=$RW_EXP@rw
RW_BETA=10
RW_EXP=8

DOMAIN=train_aug  # train2014 for COCO14
rw_inference
DOMAIN=val
rw_inference
