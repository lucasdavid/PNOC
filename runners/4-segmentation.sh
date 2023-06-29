#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH -p sequana_gpu_shared
#SBATCH -J segm
#SBATCH -o /scratch/lerdl/lucas.david/logs/segm-%j.out
#SBATCH --time=24:00:00

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
# Segmentation with Pseudo Semantic Segmentation Masks
#

# export OMP_NUM_THREADS=4

# Environment

## Local
PY=python
DEVICES=0
WORKERS_TRAIN=8
WORKERS_INFER=8
WORK_DIR=/home/ldavid/workspace/repos/research/pnoc
DATA_DIR=/home/ldavid/workspace/datasets

### Sdumont
nodeset -e $SLURM_JOB_NODELIST
module load sequana/current
module load gcc/7.4_sequana python/3.8.2_sequana cudnn/8.2_cuda-11.1_sequana
PY=python3.8
DEVICES=0,1,2,3
WORKERS_TRAIN=8
WORKERS_INFER=48
WORK_DIR=$SCRATCH/PuzzleCAM
DATA_DIR=$SCRATCH/datasets

cd $WORK_DIR
export PYTHONPATH=$(pwd)

# Dataset
## Pascal VOC 2012
DATASET=voc12
DOMAIN=train_aug
DATA_DIR=$DATA_DIR/VOCdevkit/VOC2012

IMAGE_SIZE=512
MIN_IMAGE_SIZE=256
MAX_IMAGE_SIZE=1024

### MS COCO 2014
# DATASET=coco14
# DOMAIN=train2014
# DATA_DIR=$DATA_DIR/coco14

# IMAGE_SIZE=640
# MIN_IMAGE_SIZE=320
# MAX_IMAGE_SIZE=1024

# Architecture
ARCH=rs269
ARCHITECTURE=resnest269
GROUP_NORM=true
DILATED=false
MODE=normal

LR=0.007

EPOCHS=50
BATCH_SIZE=32
ACCUMULATE_STEPS=1

AUGMENT=colorjitter # colorjitter_randaug_cutmix_mixup_cutormixup
CUTMIX=0.5
MIXUP=1.
LABELSMOOTHING=0 # 0.1

# Infrastructure
MIXED_PRECISION=true # false


make_pseudo_labels() {
  $PY scripts/segmentation/make_pseudo_labels.py \
    --experiment_name $RW_MASKS \
    --domain $DOMAIN \
    --threshold $THRESHOLD \
    --crf_t $CRF_T \
    --crf_gt_prob $CRF_GT \
    --data_dir $DATA_DIR \
    --num_workers $WORKERS_INFER
}

segm_training() {
  echo "================================================="
  echo "Semantic Segmentation Training $TAG"
  echo "================================================="

  WANDB_TAGS="$DATASET,$ARCH,segmentation,b:$BATCH_SIZE,gn,lr:$LR,ls:$LABELSMOOTHING" \
  WANDB_RUN_GROUP="$DATASET-$ARCH-segmentation" \
  CUDA_VISIBLE_DEVICES=$DEVICES \
  $PY scripts/segmentation/train.py \
    --tag $TAG \
    --lr $LR \
    --max_epoch $EPOCHS \
    --batch_size $BATCH_SIZE \
    --accumulate_steps $ACCUMULATE_STEPS \
    --mixed_precision $MIXED_PRECISION \
    --architecture $ARCHITECTURE \
    --dilated $DILATED \
    --mode $MODE \
    --use_gn $GROUP_NORM \
    --image_size $IMAGE_SIZE \
    --min_image_size $MIN_IMAGE_SIZE \
    --max_image_size $MAX_IMAGE_SIZE \
    --augment $AUGMENT \
    --cutmix_prob $CUTMIX \
    --mixup_prob $MIXUP \
    --label_smoothing $LABELSMOOTHING \
    --dataset $DATASET \
    --data_dir $DATA_DIR \
    --masks_dir $RW_MASKS_DIR \
    --num_workers $WORKERS_TRAIN
}

segm_inference() {
  echo "================================================="
  echo "Semantic Segmentation Inference $TAG"
  echo "================================================="

  CUDA_VISIBLE_DEVICES=$DEVICES \
    $PY scripts/segmentation/inference.py \
    --tag $TAG \
    --pred_dir $SEGM_PRED_DIR \
    --backbone $ARCHITECTURE \
    --mode $MODE \
    --dilated $DILATED \
    --use_gn $GROUP_NORM \
    --dataset $DATASET \
    --domain $DOMAIN \
    --crf_t $CRF_T \
    --crf_gt_prob $CRF_GT \
    --data_dir $DATA_DIR \
    --num_workers $WORKERS_TRAIN
}

run_evaluation() {
  CUDA_VISIBLE_DEVICES="" \
  WANDB_RUN_GROUP="$W_GROUP" \
  WANDB_TAGS="$W_TAGS" \
    $PY scripts/evaluate.py \
    --experiment_name $TAG  \
    --dataset $DATASET \
    --domain $DOMAIN \
    --data_dir $DATA_DIR \
    --min_th $MIN_TH \
    --max_th $MAX_TH \
    --crf_t $CRF_T \
    --crf_gt_prob $CRF_GT \
    --mode $EVAL_MODE \
    --num_workers $WORKERS_INFER
}


## 4.1 Make Pseudo Masks
##

PRIORS_TAG=ra-oc-p-poc-pnoc-avg
# PRIORS_TAG=ra-oc-p-poc-pnoc-learned-a0.25

THRESHOLD=0.3
CRF_T=1
CRF_GT=0.9

AFF_TAG=rw/$DATASET-an@ccamh@$PRIORS_TAG

DOMAIN=train_aug RW_MASKS=$AFF_TAG@train@beta=10@exp_times=8@rw make_pseudo_labels
DOMAIN=val       RW_MASKS=$AFF_TAG@val@beta=10@exp_times=8@rw   make_pseudo_labels

# Move everything (train/val) into a single folder.
RW_MASKS_DIR=./experiments/predictions/$AFF_TAG@beta=10@exp_times=8@rw@crf=$CRF_T
mv ./experiments/predictions/$AFF_TAG@train@beta=10@exp_times=8@rw@crf=$CRF_T $RW_MASKS_DIR
mv ./experiments/predictions/$AFF_TAG@val@beta=10@exp_times=8@rw@crf=$CRF_T/* $RW_MASKS_DIR/
rm -r ./experiments/predictions/$AFF_TAG@val@beta=10@exp_times=8@rw@crf=$CRF_T

## 4.2 DeepLabV3+ Training
##
LABELSMOOTHING=0.1
TAG=segmentation/$DATASET-d3p-lr$LR-ls$LABELSMOOTHING@pn-ccamh@$PRIORS_TAG
# TAG=segmentation/d3p-normal-gn-sup
# TAG=segmentation/d3p-ls0.1@pn-ccamh@rs269pnoc-ls0.1
# TAG=segmentation/coco14-d3p-lr0.004-ls0.1@pn-ccamh@rs269pnoc-lr0.05

segm_training

## 4.3 DeepLabV3+ Inference
##

CRF_T=0 DOMAIN=train SEGM_PRED_DIR=./experiments/predictions/$TAG      segm_inference
CRF_T=0 DOMAIN=val   SEGM_PRED_DIR=./experiments/predictions/$TAG      segm_inference
CRF_T=1 DOMAIN=test  SEGM_PRED_DIR=./experiments/predictions/$TAG@test segm_inference


## 4.4. Evaluation
##
EVAL_MODE=png

CRF_T=1
CRF_GT=0.9
MIN_TH=0.05
MAX_TH=0.81

DOMAIN=val
W_TAGS="$DATASET,domain:$DOMAIN,$ARCH,ensemble,ccamh,rw,segmentation,crf:$CRF_T-$CRF_GT"
run_evaluation
