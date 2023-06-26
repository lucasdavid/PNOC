#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH -p sequana_gpu_shared
#SBATCH -J ccamh
#SBATCH -o /scratch/lerdl/lucas.david/logs/ccam/tr-ccamh-%j.out
#SBATCH --time=04:00:00

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
# Train CCAM to perform the unsupervised saliency
# detection task over the VOC12 or COCO14 dataset.
# Hints from CAMs are used as additional information.
#

echo "[sdumont/sequana/saliency/train-ccam-hints] started running at $(date +'%Y-%m-%d %H:%M:%S')."

nodeset -e $SLURM_JOB_NODELIST


module load sequana/current
# module load gcc/7.4_sequana python/3.9.1_sequana cudnn/8.2_cuda-11.1_sequana
# module load gcc/7.4 python/3.9.1 cudnn/8.2_cuda-11.1
module load gcc/7.4_sequana python/3.8.2_sequana cudnn/8.2_cuda-11.1_sequana

PRJ_DIR=$SCRATCH/PuzzleCAM

cd $PRJ_DIR

export PYTHONPATH=$(pwd)

# PY=python3.9
PY=python3.8
DEVICES=0,1,2,3

DATASET=voc12
DATA_DIR=$SCRATCH/datasets/VOCdevkit/VOC2012/
DOMAIN=train
# DATASET=coco14
# DATA_DIR=$SCRATCH/datasets/coco14/
# DOMAIN=train2014

LOGS_DIR=$SCRATCH/logs/ccam

WORKERS=8

IMAGE_SIZE=448
EPOCHS=10
BATCH_SIZE=32
ACCUMULATE_STEPS=1
MIXED_PRECISION=true
LABELSMOOTHING=0

ARCHITECTURE=resnest269
ARCH=rs269
DILATED=false
TRAINABLE_STEM=true
MODE=normal
S4_OUT_FEATURES=1024

ALPHA=0.25
HINT_W=1.0
LR=0.001

FG_T=0.4
# BG_T=0.1

INF_FG_T=0.2
CRF_T=10
CRF_GT_PROB=0.7

ccamh_training() {
  WANDB_TAGS="ccamh,amp,$DATASET,$ARCH,b:$BATCH_SIZE,ac:$ACCUMULATE_STEPS,fg:$FG_T,ls:$LABELSMOOTHING,lr:$LR"     \
  CUDA_VISIBLE_DEVICES=$DEVICES $PY scripts/ccam/train_hints.py \
    --tag             $TAG                  \
    --alpha           $ALPHA                \
    --hint_w          $HINT_W               \
    --max_epoch       $EPOCHS               \
    --batch_size      $BATCH_SIZE           \
    --lr              $LR                   \
    --label_smoothing $LABELSMOOTHING       \
    --accumulate_steps $ACCUMULATE_STEPS    \
    --mixed_precision $MIXED_PRECISION      \
    --num_workers     $WORKERS              \
    --architecture    $ARCHITECTURE         \
    --stage4_out_features $S4_OUT_FEATURES  \
    --dilated         $DILATED              \
    --mode            $MODE                 \
    --trainable-stem  $TRAINABLE_STEM       \
    --image_size      $IMAGE_SIZE           \
    --cams_dir        $CAMS_DIR             \
    --fg_threshold    $FG_T                 \
    --dataset         $DATASET              \
    --data_dir        $DATA_DIR
    # --bg_threshold    $BG_T                \
}


ccamh_inference() {
  WEIGHTS=imagenet
  PRETRAINED=./experiments/models/$TAG.pth

  CUDA_VISIBLE_DEVICES=$DEVICES $PY scripts/ccam/inference.py \
    --tag             $TAG                 \
    --dataset         $DATASET             \
    --domain          $DOMAIN              \
    --architecture    $ARCHITECTURE        \
    --dilated         $DILATED             \
    --stage4_out_features $S4_OUT_FEATURES \
    --mode            $MODE                \
    --weights         $WEIGHTS             \
    --trainable-stem  $TRAINABLE_STEM      \
    --pretrained      $PRETRAINED          \
    --data_dir        $DATA_DIR
}

ccamh_pseudo_masks_crf() {
  $PY scripts/ccam/inference_crf.py                    \
    --experiment_name $TAG@train@scale=0.5,1.0,1.5,2.0 \
    --dataset         $DATASET                         \
    --domain          $DOMAIN                          \
    --threshold       $INF_FG_T                        \
    --crf_t           $CRF_T                           \
    --crf_gt_prob     $CRF_GT_PROB                     \
    --data_dir        $DATA_DIR
}

poolnet_training() {
  cd $PRJ_DIR/poolnet

  CUDA_VISIBLE_DEVICES=0 $PY main_custom.py  \
    --arch        "resnet"       \
    --mode        "train"             \
    --dataset     $DATASET            \
    --train_root  $DATA_DIR           \
    --train_list  $DOMAIN             \
    --pseudo_root $PSEUDO_CUES
  
  cd $PRJ_DIR
}

poolnet_inference() {
  cd $PRJ_DIR/poolnet

  CUDA_VISIBLE_DEVICES=0 $PY main_custom.py  \
    --arch        "resnet"       \
    --mode        "test"              \
    --model       $MODEL_CKPT         \
    --dataset     $DATASET            \
    --train_list  $DOMAIN             \
    --train_root  $DATA_DIR           \
    --pseudo_root $PSEUDO_CUES        \
    --sal_folder ./results/$TAG

  cd $PRJ_DIR
}


## CCAM-H Training and Inference
## =============================

# CAMS_DIR=$PRJ_DIR/experiments/predictions/resnest101@randaug@train@scale=0.5,1.0,1.5,2.0
# TAG=saliency/$DATASET-ccamh-$ARCH@rs269poc@rs101ra@b$BATCH_SIZE-fg$FG_T
# ccamh_training
# ccamh_inference

# CAMS_DIR=$PRJ_DIR/experiments/predictions/poc/ResNeSt269@PuzzleOc@train@scale=0.5,1.0,1.5,2.0
# TAG=saliency/$DATASET-ccamh-$ARCH@rs269poc@fg$FG_T-h$HINT_W-e$EPOCHS-b$BATCH_SIZE-lr$LR
# ccamh_training
# ccamh_inference

# CAMS_DIR=$PRJ_DIR/experiments/predictions/poc/voc12-rs269-poc-ls0.1@rs269ra-r3@train@scale=0.5,1.0,1.5,2.0
# TAG=saliency/$DATASET-ccamh-$ARCH@rs269poc-ls0.1-r3@fg$FG_T-h$HINT_W-e$EPOCHS-b$BATCH_SIZE-lr$LR
# ccamh_training
# ccamh_inference

# CAMS_DIR=$PRJ_DIR/experiments/predictions/poc/voc12-rs269-poc-ls0.1@rs269ra-r3@train@scale=0.5,1.0,1.5,2.0
# LABELSMOOTHING=0.1
# TAG=saliency/$DATASET-ccamh-$ARCH@rs269poc-ls0.1-r3@fg$FG_T-ls$LABELSMOOTHING-h$HINT_W-e$EPOCHS-b$BATCH_SIZE-lr$LR
# ccamh_training
# ccamh_inference

# BATCH_SIZE=64
# CAMS_DIR=$PRJ_DIR/experiments/predictions/poc/voc12-rs269-poc-ls0.1@rs269ra-r3@train@scale=0.5,1.0,1.5,2.0
# TAG=saliency/$DATASET-ccamh-$ARCH@rs269poc-ls0.1-r3@fg$FG_T-h$HINT_W-e$EPOCHS-b$BATCH_SIZE-lr$LR-amp
# ccamh_training
# ccamh_inference

# CAMS_DIR=$PRJ_DIR/experiments/predictions/pnoc/voc12-rs269-pnoc-ls0.1-ow0.0-1.0-1.0-cams-0.2-octis1-amp@rs269ra-r3@train@scale=0.5,1.0,1.5,2.0
# TAG=saliency/$DATASET-ccamh-$ARCH@rs269pnoc-ls0.1-r3@fg$FG_T-h$HINT_W-e$EPOCHS-b$BATCH_SIZE-lr$LR
# ccamh_training
# ccamh_inference

# FG_T=0.3
# CAMS_DIR=$PRJ_DIR/experiments/predictions/pnoc/voc12-rs269-pnoc-ls0.1-ow0.0-1.0-1.0-cams-0.2-octis1-amp@rs269ra-r3@train@scale=0.5,1.0,1.5,2.0
# TAG=saliency/$DATASET-ccamh-$ARCH@rs269pnoc-ls0.1-r3@fg$FG_T-h$HINT_W-e$EPOCHS-b$BATCH_SIZE-lr$LR
# ccamh_training
# ccamh_inference

# FG_T=0.25
# CAMS_DIR=$PRJ_DIR/experiments/predictions/pnoc/voc12-rs269-pnoc-ls0.1-ow0.0-1.0-1.0-cams-0.2-octis1-amp@rs269ra-r3@train@scale=0.5,1.0,1.5,2.0
# TAG=saliency/$DATASET-ccamh-$ARCH@rs269pnoc-ls0.1-r3@fg$FG_T-h$HINT_W-e$EPOCHS-b$BATCH_SIZE-lr$LR
# ccamh_training
# ccamh_inference


# FG_T=0.2
# CAMS_DIR=$PRJ_DIR/experiments/predictions/pnoc/voc12-rs269-pnoc-ls0.1-ow0.0-1.0-1.0-cams-0.2-octis1-amp@rs269ra-r3@train@scale=0.5,1.0,1.5,2.0
# TAG=saliency/$DATASET-ccamh-$ARCH@rs269pnoc-ls0.1-r3@fg$FG_T-h$HINT_W-e$EPOCHS-b$BATCH_SIZE-lr$LR-amp
# ccamh_training
# ccamh_inference
# ccamh_pseudo_masks_crf
