#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH -p sequana_gpu_shared
#SBATCH -J tr-aff
#SBATCH -o /scratch/lerdl/lucas.david/logs/aff-%j.out
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
# Train a model to perform multilabel classification over
# a weakly supervisedsemantic segmentation dataset.
#

# export OMP_NUM_THREADS=4

# Environment

## Local
PY=python
DEVICE="cuda"
DEVICES="0"
WORKERS_TRAIN=4
WORKERS_INFER=4
WORK_DIR=/home/ldavid/workspace/repos/research/pnoc
DATA_DIR=/home/ldavid/workspace/datasets

#### Sdumont
# nodeset -e $SLURM_JOB_NODELIST
# module load sequana/current
# module load gcc/7.4_sequana python/3.8.2_sequana cudnn/8.2_cuda-11.1_sequana
# PY=python3.8
# DEVICE="cuda"
# DEVICES=0,1,2,3
# WORKERS_TRAIN=8
# WORKERS_INFER=48
# WORK_DIR=$SCRATCH/PuzzleCAM
# DATA_DIR=$SCRATCH/datasets

cd $WORK_DIR
export PYTHONPATH=$(pwd)

# Dataset
## Pascal VOC 2012
# DATASET=voc12
# DOMAIN=train_aug
# DATA_DIR=$DATA_DIR/VOCdevkit/VOC2012

# IMAGE_SIZE=320
# MIN_IMAGE_SIZE=300
# MAX_IMAGE_SIZE=340

### MS COCO 2014
DATASET=coco14
DOMAIN=train2014
DATA_DIR=$DATA_DIR/coco14

IMAGE_SIZE=320
MIN_IMAGE_SIZE=300
MAX_IMAGE_SIZE=340

## DeepGlobe Land Cover Classification
# DATASET=deepglobe
# DOMAIN=train75
# DATA_DIR=$DATA_DIR/DGdevkit

# IMAGE_SIZE=320
# MIN_IMAGE_SIZE=300
# MAX_IMAGE_SIZE=340

# Architecture
ARCH=rn50
ARCHITECTURE=resnet50
TRAINABLE_STEM=true
DILATED=false
MODE=normal
REGULAR=none

CUTMIX=0.5
MIXUP=1.0
LABELSMOOTHING=0 # 0.1

EPOCHS=15
BATCH_SIZE=32
LR=0.1

PERFORM_VALIDATION=false

train_vanilla() {
  echo "=================================================================="
  echo "[train vanilla:$TAG_VANILLA] started at $(date +'%Y-%m-%d %H:%M:%S')."
  echo "=================================================================="

  WANDB_TAGS="$DATASET,$ARCH,lr:$LR,ls:$LABELSMOOTHING,b:$BATCH_SIZE" \
    WANDB_RUN_GROUP="$DATASET-$ARCH-vanilla" \
    CUDA_VISIBLE_DEVICES=$DEVICES \
    $PY scripts/cam/train_vanilla.py \
    --tag $TAG_VANILLA \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --architecture $ARCHITECTURE \
    --dilated $DILATED \
    --mode $MODE \
    --trainable-stem $TRAINABLE_STEM \
    --regularization $REGULAR \
    --image_size $IMAGE_SIZE \
    --min_image_size $MIN_IMAGE_SIZE \
    --max_image_size $MAX_IMAGE_SIZE \
    --augment $AUGMENT \
    --cutmix_prob $CUTMIX \
    --mixup_prob $MIXUP \
    --label_smoothing $LABELSMOOTHING \
    --max_epoch $EPOCHS \
    --dataset $DATASET \
    --data_dir $DATA_DIR \
    --validate $PERFORM_VALIDATION \
    --device $DEVICE \
    --num_workers $WORKERS_TRAIN
}

train_poc() {
  echo "=================================================================="
  echo "[train vanilla:$TAG] started at $(date +'%Y-%m-%d %H:%M:%S')."
  echo "=================================================================="

  WANDB_TAGS="$DATASET,$ARCH,lr:$LR,ls:$LABELSMOOTHING,b:$BATCH_SIZE,poc" \
    WANDB_RUN_GROUP="$DATASET-$ARCH-poc" \
    CUDA_VISIBLE_DEVICES=$DEVICES \
    $PY scripts/cam/train_poc.py \
    --tag $TAG \
    --num_workers $WORKERS_TRAIN \
    --batch_size $BATCH_SIZE \
    --architecture $ARCHITECTURE \
    --dilated $DILATED \
    --mode $MODE \
    --trainable-stem $TRAINABLE_STEM \
    --regularization $REGULAR \
    --oc-architecture $OC_ARCHITECTURE \
    --oc-pretrained $OC_PRETRAINED \
    --oc-regularization $OC_REGULAR \
    --oc-mask-globalnorm $OC_MASK_GN \
    --image_size $IMAGE_SIZE \
    --min_image_size $MIN_IMAGE_SIZE \
    --max_image_size $MAX_IMAGE_SIZE \
    --augment $AUGMENT \
    --cutmix_prob $CUTMIX \
    --mixup_prob $MIXUP \
    --label_smoothing $LABELSMOOTHING \
    --max_epoch $EPOCHS \
    --alpha $P_ALPHA \
    --alpha_init $P_INIT \
    --alpha_schedule $P_SCHEDULE \
    --oc-alpha $OC_ALPHA \
    --oc-alpha-init $OC_INIT \
    --oc-alpha-schedule $OC_SCHEDULE \
    --oc-strategy $OC_STRATEGY \
    --oc-focal-momentum $OC_F_MOMENTUM \
    --oc-focal-gamma $OC_F_GAMMA \
    --dataset $DATASET \
    --data_dir $DATA_DIR
}

train_puzzle() {
  echo "=================================================================="
  echo "[train pnoc:$TAG] started at $(date +'%Y-%m-%d %H:%M:%S')."
  echo "=================================================================="

  CUDA_VISIBLE_DEVICES=$DEVICES \
    WANDB_TAGS="$DATASET,$ARCH,lr:$LR,ls:$LABELSMOOTHING,b:$BATCH_SIZE,puzzle" \
    WANDB_RUN_GROUP=$DATASET-$ARCH-p \
    $PY scripts/cam/train_puzzle.py \
    --tag $TAG \
    --lr $LR \
    --num_workers $WORKERS_TRAIN \
    --max_epoch $EPOCHS \
    --batch_size $BATCH_SIZE \
    --architecture $ARCHITECTURE \
    --dilated $DILATED \
    --mode $MODE \
    --trainable-stem $TRAINABLE_STEM \
    --regularization $REGULAR \
    --re_loss_option masking \
    --re_loss L1_Loss \
    --alpha_schedule 0.50 \
    --alpha 4.00 \
    --image_size $IMAGE_SIZE \
    --min_image_size $MIN_IMAGE_SIZE \
    --max_image_size $MAX_IMAGE_SIZE \
    --augment $AUGMENT \
    --cutmix_prob $CUTMIX \
    --mixup_prob $MIXUP \
    --label_smoothing $LABELSMOOTHING \
    --dataset $DATASET \
    --data_dir $DATA_DIR
}

train_pnoc() {
  echo "=================================================================="
  echo "[train pnoc:$TAG] started at $(date +'%Y-%m-%d %H:%M:%S')."
  echo "=================================================================="

  CUDA_VISIBLE_DEVICES=$DEVICES \
    WANDB_TAGS="$DATASET,$ARCH,lr:$LR,ls:$LABELSMOOTHING,b:$BATCH_SIZE,ac:$ACCUMULATE_STEPS,pnoc" \
    WANDB_RUN_GROUP=$DATASET-$ARCH-pnoc-ow$OW_INIT-$OW-$OW_SCHEDULE-c$OC_TRAIN_MASK_T \
    $PY scripts/cam/train_pnoc.py \
    --tag $TAG \
    --num_workers $WORKERS_TRAIN \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --accumulate_steps $ACCUMULATE_STEPS \
    --mixed_precision $MIXED_PRECISION \
    --architecture $ARCHITECTURE \
    --dilated $DILATED \
    --mode $MODE \
    --trainable-stem $TRAINABLE_STEM \
    --regularization $REGULAR \
    --oc-architecture $OC_ARCHITECTURE \
    --oc-pretrained $OC_PRETRAINED \
    --oc-regularization $OC_REGULAR \
    --oc-trainable-stem $OC_TRAINABLE_STEM \
    --image_size $IMAGE_SIZE \
    --min_image_size $MIN_IMAGE_SIZE \
    --max_image_size $MAX_IMAGE_SIZE \
    --augment $AUGMENT \
    --cutmix_prob $CUTMIX \
    --mixup_prob $MIXUP \
    --label_smoothing $LABELSMOOTHING \
    --max_epoch $EPOCHS \
    --alpha $P_ALPHA \
    --alpha_init $P_INIT \
    --alpha_schedule $P_SCHEDULE \
    --oc-alpha $OC_ALPHA \
    --oc-alpha-init $OC_INIT \
    --oc-alpha-schedule $OC_SCHEDULE \
    --oc-strategy $OC_STRATEGY \
    --ow $OW \
    --ow-init $OW_INIT \
    --ow-schedule $OW_SCHEDULE \
    --oc-train-masks $OC_TRAIN_MASKS \
    --oc_train_mask_t $OC_TRAIN_MASK_T \
    --oc-train-interval-steps $OC_TRAIN_INT_STEPS \
    --validate $PERFORM_VALIDATION \
    --dataset $DATASET \
    --data_dir $DATA_DIR
}

inference_priors() {
  echo "=================================================================="
  echo "[Inference:$TAG] started at $(date +'%Y-%m-%d %H:%M:%S')."
  echo "=================================================================="

  CUDA_VISIBLE_DEVICES=$DEVICES \
    $PY scripts/cam/inference.py \
    --architecture $ARCHITECTURE \
    --regularization $REGULAR \
    --dilated $DILATED \
    --trainable-stem $TRAINABLE_STEM \
    --mode $MODE \
    --tag $TAG \
    --domain $DOMAIN \
    --data_dir $DATA_DIR \
    --device $DEVICE
}

# MODE=fix
# LR=0.05
# TRAINABLE_STEM=false

AUGMENT=randaugment
LABELSMOOTHING=0.1
TAG_VANILLA=vanilla/$DATASET-$ARCH-ra-ls-fix-lr$LR

train_vanilla

MIXED_PRECISION=true
BATCH_SIZE=16
ACCUMULATE_STEPS=2

OC_NAME="$ARCH"-rals
OC_PRETRAINED=experiments/models/$TAG_VANILLA.pth
OC_ARCHITECTURE=$ARCHITECTURE
OC_REGULAR=none
OC_TRAINABLE_STEM=true
OC_STRATEGY=random

# Schedule
P_INIT=0.0
P_ALPHA=4.0
P_SCHEDULE=0.5

OC_INIT=0.3
OC_ALPHA=1.0
OC_SCHEDULE=1.0

OW=1.0
OW_INIT=0.0
OW_SCHEDULE=0.5
OC_TRAIN_MASKS=cams
OC_TRAIN_MASK_T=0.2
OC_TRAIN_INT_STEPS=1

AUGMENT=colorjitter
LABELSMOOTHING=0.1

TAG="pnoc/$DATASET-$ARCH-pnoc-b$BATCH_SIZE-a$ACCUMULATE_STEPS-ls$LABELSMOOTHING@$OC_NAME-r10"
# train_pnoc
# inference_priors
