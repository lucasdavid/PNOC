#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH -p sequana_gpu_shared
#SBATCH -J priors
#SBATCH -o /scratch/lerdl/lucas.david/logs/%j-priors.out
#SBATCH --time=24:00:00

# Copyright 2023 Lucas Oliveira David
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
# Train a model to perform multilabel classification over a WSSS dataset.
#

if [[ "`hostname`" == "sdumont"* ]]; then
  ENV=sdumont
  WORK_DIR=$SCRATCH/pnoc
else
  ENV=local
  WORK_DIR=$HOME/workspace/repos/research/wsss/pnoc
fi

## Dataset
# DATASET=voc12       # Pascal VOC 2012
# DATASET=coco14      # MS COCO 2014
# DATASET=deepglobe   # DeepGlobe Land Cover Classification
# DATASET=cityscapes  # Cityscapes Urban Semantic Segmentation
DATASET=hpa-single-cell-classification  # HPA Single Cell Classification

. $WORK_DIR/runners/config/env.sh
. $WORK_DIR/runners/config/dataset.sh

cd $WORK_DIR
export PYTHONPATH=$(pwd)

## Architecture
### Priors

ARCHITECTURE=resnest101
ARCH=rs101

# ARCH=rs269
# ARCHITECTURE=resnest269

# ARCHITECTURE=swin_b
# ARCH=swin_b_22k
# PRETRAINED_WEIGHTS=./experiments/models/pretrained/swin_base_patch4_window7_224_22k.pth

# ARCHITECTURE=swin_l
# ARCH=swin_l_22k
# PRETRAINED_WEIGHTS=./experiments/models/pretrained/swin_large_patch4_window7_224_22k.pth

# ARCHITECTURE=mit_b0
# ARCH=mit_b0
# PRETRAINED_WEIGHTS=./experiments/models/pretrained/mit_b0.pth

# ARCHITECTURE=mit_b5
# ARCH=mit_b5
# PRETRAINED_WEIGHTS=./experiments/models/pretrained/mit_b5.pth

TRAINABLE_STEM=true
TRAINABLE_STAGE4=true
TRAINABLE_BONE=true
DILATED=false
MODE=normal
PRETRAINED_WEIGHTS=imagenet

# Training
OPTIMIZER=sgd  # sgd,lion
EPOCHS=15
BATCH=16
ACCUMULATE_STEPS=1

LR_ALPHA_SCRATCH=10.0
LR_ALPHA_BIAS=2.0
LR_ALPHA_OC=1.0

# =========================
# $PIP install lion-pytorch
# OPTIMIZER=lion
# LR_ALPHA_SCRATCH=1.0
# LR_ALPHA_BIAS=1.0
# LR=0.00001
# WD=0.01
# =========================

EMA_ENABLED=false
EMA_WARMUP=128
EMA_STEPS=32

MIXED_PRECISION=true
PERFORM_VALIDATION=true
VALIDATE_PRIORS=false

## Augmentation
AUGMENT=randaugment  # clahe_collorjitter_mixup_cutmix_cutormixup
AUG=ra
CUTMIX=0.5
MIXUP=1.0
LABELSMOOTHING=0

## OC-CSE
OC_ARCHITECTURE=$ARCHITECTURE
OC_MASK_GN=true # originally done in OC-CSE
OC_STRATEGY=random
OC_F_MOMENTUM=0.9
OC_F_GAMMA=2.0
OC_PERSIST=false

## Schedule
P_INIT=0.0
P_ALPHA=4.0
P_SCHEDULE=0.5

OC_INIT=0.3
OC_ALPHA=1.0
OC_SCHEDULE=1.0

OW=1.0
OW_INIT=0.0
OW_SCHEDULE=1.0
OC_TRAIN_MASKS=cams
OC_TRAIN_MASK_T=0.2
OC_TRAIN_INT_STEPS=1

# Evaluation
MIN_TH=0.05
MAX_TH=0.81
CRF_T=0
CRF_GT=0.7


train_vanilla() {
  echo "=================================================================="
  echo "[train $TAG_VANILLA] started at $(date +'%Y-%m-%d %H:%M:%S')."
  echo "=================================================================="

  WANDB_TAGS="$DATASET,$ARCH,lr:$LR,wd:$WD,ls:$LABELSMOOTHING,b:$BATCH,aug:$AUG" \
    WANDB_RUN_GROUP="$DATASET-$ARCH-vanilla" \
    CUDA_VISIBLE_DEVICES=$DEVICES \
    $PY scripts/cam/train_vanilla.py \
    --tag $TAG_VANILLA \
    --lr $LR \
    --wd $WD \
    --optimizer $OPTIMIZER \
    --lr_alpha_scratch $LR_ALPHA_SCRATCH \
    --lr_alpha_bias $LR_ALPHA_BIAS \
    --batch_size $BATCH \
    --accumulate_steps $ACCUMULATE_STEPS \
    --ema $EMA_ENABLED \
    --ema_warmup $EMA_WARMUP \
    --ema_steps $EMA_STEPS \
    --mixed_precision $MIXED_PRECISION \
    --architecture $ARCHITECTURE \
    --dilated $DILATED \
    --mode $MODE \
    --backbone_weights $PRETRAINED_WEIGHTS \
    --trainable-stem $TRAINABLE_STEM \
    --trainable-backbone $TRAINABLE_BONE \
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
    --domain_train $DOMAIN_TRAIN \
    --domain_valid $DOMAIN_VALID \
    --validate $PERFORM_VALIDATION \
    --validate_priors $VALIDATE_PRIORS \
    --validate_max_steps $VALIDATE_MAX_STEPS \
    --validate_thresholds $VALIDATE_THRESHOLDS \
    --device $DEVICE \
    --num_workers $WORKERS_TRAIN
}

train_occse() {
  echo "=================================================================="
  echo "[train $TAG] started at $(date +'%Y-%m-%d %H:%M:%S')."
  echo "=================================================================="

  # default OC-CSE parameters <https://github.com/KAIST-vilab/OC-CSE>
  MODE=fix
  TRAINABLE_STEM=false
  DILATED=true

  WANDB_TAGS="$DATASET,$ARCH,lr:$LR,ls:$LABELSMOOTHING,b:$BATCH,ac:$ACCUMULATE_STEPS,oc-cse" \
    WANDB_RUN_GROUP="$DATASET-$ARCH-occse" \
    CUDA_VISIBLE_DEVICES=$DEVICES \
    $PY scripts/cam/train_occse.py \
    --tag $TAG \
    --lr $LR \
    --num_workers $WORKERS_TRAIN \
    --batch_size $BATCH \
    --accumulate_steps $ACCUMULATE_STEPS \
    --mixed_precision $MIXED_PRECISION \
    --architecture $ARCHITECTURE \
    --dilated $DILATED \
    --mode "$MODE" \
    --trainable-stem $TRAINABLE_STEM \
    --oc-architecture $OC_ARCHITECTURE \
    --oc-pretrained $OC_PRETRAINED \
    --oc-mask-globalnorm $OC_MASK_GN \
    --image_size $IMAGE_SIZE \
    --min_image_size $MIN_IMAGE_SIZE \
    --max_image_size $MAX_IMAGE_SIZE \
    --augment $AUGMENT \
    --cutmix_prob $CUTMIX \
    --mixup_prob $MIXUP \
    --label_smoothing $LABELSMOOTHING \
    --max_epoch $EPOCHS \
    --oc-alpha $OC_ALPHA \
    --oc-alpha-init $OC_INIT \
    --oc-alpha-schedule $OC_SCHEDULE \
    --oc-strategy $OC_STRATEGY \
    --oc-focal-momentum $OC_F_MOMENTUM \
    --oc-focal-gamma $OC_F_GAMMA \
    --validate $PERFORM_VALIDATION \
    --validate_max_steps $VALIDATE_MAX_STEPS \
    --validate_thresholds $VALIDATE_THRESHOLDS \
    --dataset $DATASET \
    --domain_train $DOMAIN_TRAIN \
    --domain_valid $DOMAIN_VALID \
    --data_dir $DATA_DIR;
}

train_puzzle() {
  echo "=================================================================="
  echo "[train $TAG] started at $(date +'%Y-%m-%d %H:%M:%S')."
  echo "=================================================================="

  CUDA_VISIBLE_DEVICES=$DEVICES \
    WANDB_TAGS="$DATASET,$ARCH,lr:$LR,ls:$LABELSMOOTHING,b:$BATCH,ac:$ACCUMULATE_STEPS,puzzle" \
    WANDB_RUN_GROUP=$DATASET-$ARCH-p \
    $PY scripts/cam/train_puzzle.py \
    --tag $TAG \
    --lr $LR \
    --num_workers $WORKERS_TRAIN \
    --max_epoch $EPOCHS \
    --batch_size $BATCH \
    --accumulate_steps $ACCUMULATE_STEPS \
    --mixed_precision $MIXED_PRECISION \
    --architecture $ARCHITECTURE \
    --dilated $DILATED \
    --mode $MODE \
    --trainable-stem $TRAINABLE_STEM \
    --trainable-backbone $TRAINABLE_BONE \
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
    --validate $PERFORM_VALIDATION \
    --validate_max_steps $VALIDATE_MAX_STEPS \
    --validate_thresholds $VALIDATE_THRESHOLDS \
    --dataset $DATASET \
    --domain_train $DOMAIN_TRAIN \
    --domain_valid $DOMAIN_VALID \
    --data_dir $DATA_DIR
}

train_poc() {
  echo "=================================================================="
  echo "[train $TAG] started at $(date +'%Y-%m-%d %H:%M:%S')."
  echo "=================================================================="

  WANDB_TAGS="$DATASET,$ARCH,lr:$LR,ls:$LABELSMOOTHING,b:$BATCH,ac:$ACCUMULATE_STEPS,poc" \
    WANDB_RUN_GROUP="$DATASET-$ARCH-poc" \
    CUDA_VISIBLE_DEVICES=$DEVICES \
    $PY scripts/cam/train_poc.py \
    --tag $TAG \
    --lr $LR \
    --num_workers $WORKERS_TRAIN \
    --batch_size $BATCH \
    --accumulate_steps $ACCUMULATE_STEPS \
    --mixed_precision $MIXED_PRECISION \
    --architecture $ARCHITECTURE \
    --dilated $DILATED \
    --mode $MODE \
    --trainable-stem $TRAINABLE_STEM \
    --trainable-backbone $TRAINABLE_BONE \
    --oc-architecture $OC_ARCHITECTURE \
    --oc-pretrained $OC_PRETRAINED \
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
    --validate $PERFORM_VALIDATION \
    --validate_max_steps $VALIDATE_MAX_STEPS \
    --validate_thresholds $VALIDATE_THRESHOLDS \
    --dataset $DATASET \
    --domain_train $DOMAIN_TRAIN \
    --domain_valid $DOMAIN_VALID \
    --data_dir $DATA_DIR;
}

train_pnoc() {
  echo "=================================================================="
  echo "[train $TAG] started at $(date +'%Y-%m-%d %H:%M:%S')."
  echo "=================================================================="

  CUDA_VISIBLE_DEVICES=$DEVICES \
    WANDB_TAGS="$DATASET,$ARCH,lr:$LR,wd:$WD,ls:$LABELSMOOTHING,b:$BATCH,ac:$ACCUMULATE_STEPS,pnoc" \
    WANDB_RUN_GROUP=$DATASET-$ARCH-pnoc-ow$OW_INIT-$OW-$OW_SCHEDULE-c$OC_TRAIN_MASK_T \
    $PY scripts/cam/train_pnoc.py \
    --tag $TAG \
    --num_workers $WORKERS_TRAIN \
    --lr $LR \
    --wd $WD \
    --lr_alpha_scratch $LR_ALPHA_SCRATCH \
    --lr_alpha_bias $LR_ALPHA_BIAS \
    --lr_alpha_oc   $LR_ALPHA_OC \
    --optimizer $OPTIMIZER \
    --batch_size $BATCH \
    --accumulate_steps $ACCUMULATE_STEPS \
    --mixed_precision $MIXED_PRECISION \
    --architecture $ARCHITECTURE \
    --dilated $DILATED \
    --mode $MODE \
    --backbone_weights $PRETRAINED_WEIGHTS \
    --trainable-stem $TRAINABLE_STEM \
    --trainable-backbone $TRAINABLE_BONE \
    --trainable-stage4 $TRAINABLE_STAGE4 \
    --oc-architecture $OC_ARCHITECTURE \
    --oc-pretrained $OC_PRETRAINED \
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
    --oc-persist $OC_PERSIST \
    --validate $PERFORM_VALIDATION \
    --validate_priors $VALIDATE_PRIORS \
    --validate_max_steps $VALIDATE_MAX_STEPS \
    --validate_thresholds $VALIDATE_THRESHOLDS \
    --dataset $DATASET \
    --domain_train $DOMAIN_TRAIN \
    --domain_valid $DOMAIN_VALID \
    --data_dir $DATA_DIR;
}

inference_priors() {
  echo "=================================================================="
  echo "[Inference:$TAG] started at $(date +'%Y-%m-%d %H:%M:%S')."
  echo "=================================================================="

  CUDA_VISIBLE_DEVICES=$DEVICES \
    $PY scripts/cam/inference.py \
    --architecture $ARCHITECTURE \
    --dilated $DILATED \
    --mode $MODE \
    --tag $TAG \
    --dataset $DATASET \
    --domain $DOMAIN \
    --resize $INF_IMAGE_SIZE \
    --data_dir $DATA_DIR \
    --device $DEVICE
}

evaluate_priors() {
  WANDB_TAGS="$DATASET,$ARCH,lr:$LR,ls:$LABELSMOOTHING,b:$BATCH,ac:$ACCUMULATE_STEPS,domain:$DOMAIN,crf:$CRF_T-$CRF_GT" \
  CUDA_VISIBLE_DEVICES="" \
  $PY scripts/evaluate.py \
    --experiment_name $TAG \
    --dataset $DATASET \
    --domain $DOMAIN \
    --data_dir $DATA_DIR \
    --min_th $MIN_TH \
    --max_th $MAX_TH \
    --crf_t $CRF_T \
    --crf_gt_prob $CRF_GT \
    --mode npy \
    --num_workers $WORKERS_INFER;
}

LABELSMOOTHING=0.1
AUGMENT=none  # default:randaugment, cityscapes:clahe
AUG=no

EID=r1  # Experiment ID

TAG=vanilla/$DATASET-$ARCH-lr$LR-$EID
TAG_VANILLA=$TAG
train_vanilla

# MODE=fix
# TRAINABLE_STAGE4=false
BATCH=16
ACCUMULATE_STEPS=2
# LABELSMOOTHING=0

AUGMENT=none  # default:colorjitter, cityscapes:clahe, deepglobe:none
AUG=no

OC_NAME="$ARCH"
OC_PRETRAINED=experiments/models/$TAG_VANILLA.pth

# TAG="puzzle/$DATASET-$ARCH-p-b$BATCH-lr$LR-$EID"
# train_puzzle

# TAG="poc/$DATASET-$ARCH-poc-b$BATCH-lr$LR-ls@$OC_NAME-$EID"
# train_poc

TAG="pnoc/$DATASET-$ARCH-pnoc-b$BATCH-lr$LR-ls@$OC_NAME-$EID"
train_pnoc

# DOMAIN=$DOMAIN_TRAIN     inference_priors
# DOMAIN=$DOMAIN_VALID     inference_priors
# DOMAIN=$DOMAIN_VALID_SEG inference_priors

# CRF_T=0  DOMAIN=$DOMAIN_VALID     TAG=$TAG@train@scale=0.5,1.0,1.5,2.0 evaluate_priors
# CRF_T=10 DOMAIN=$DOMAIN_VALID     TAG=$TAG@train@scale=0.5,1.0,1.5,2.0 evaluate_priors
# CRF_T=0  DOMAIN=$DOMAIN_VALID_SEG TAG=$TAG@val@scale=0.5,1.0,1.5,2.0   evaluate_priors
# CRF_T=10 DOMAIN=$DOMAIN_VALID_SEG TAG=$TAG@val@scale=0.5,1.0,1.5,2.0   evaluate_priors
