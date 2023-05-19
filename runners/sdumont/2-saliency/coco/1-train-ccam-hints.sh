#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH -p sequana_gpu_shared
#SBATCH -J tr-ccam
#SBATCH -o /scratch/lerdl/lucas.david/logs/ccam/tr-ccamh-coco-%j.out
#SBATCH --time=36:00:00

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

cd $SCRATCH/PuzzleCAM

module load sequana/current
module load gcc/7.4_sequana python/3.9.1_sequana cudnn/8.2_cuda-11.1_sequana
# module load gcc/7.4 python/3.9.1 cudnn/8.2_cuda-11.1


export PYTHONPATH=$(pwd)

PY=python3.9
SOURCE=scripts/ccam/train_hints.py
DEVICES=0,1,2,3

# DATASET=voc12
# DATA_DIR=$SCRATCH/datasets/VOCdevkit/VOC2012/
# DOMAIN=train
DATASET=coco14
DATA_DIR=$SCRATCH/datasets/coco14/
DOMAIN=train2014

LOGS_DIR=$SCRATCH/logs/ccam

WORKERS=8

IMAGE_SIZE=448
EPOCHS=10
BATCH_SIZE=32
ACCUMULATE_STEPS=1
MIXED_PRECISION=false
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

run_training() {
  WANDB_TAGS="ccamh,$DATASET,$ARCH,b:$BATCH_SIZE,ac:$ACCUMULATE_STEPS,fg-hints:$FG_T,lr:$LR,ls:$LABELSMOOTHING"     \
  CUDA_VISIBLE_DEVICES=$DEVICES $PY $SOURCE \
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


run_inference() {
  WEIGHTS=imagenet
  PRETRAINED=./experiments/models/$TAG.pth

  CUDA_VISIBLE_DEVICES=$DEVICES $PY scripts/ccam/inference.py \
    --tag             $TAG                 \
    --architecture    $ARCHITECTURE        \
    --dilated         $DILATED             \
    --stage4_out_features $S4_OUT_FEATURES \
    --mode            $MODE                \
    --weights         $WEIGHTS             \
    --trainable-stem  $TRAINABLE_STEM      \
    --pretrained      $PRETRAINED          \
    --dataset         $DATASET             \
    --domain          $DOMAIN              \
    --data_dir        $DATA_DIR
}

FG_T=0.4
# BG_T=0.1

# CAMS_DIR=$SCRATCH/PuzzleCAM/experiments/predictions/pnoc/coco14-rs269-pnoc-b16-a2-ls0.1-ow0.0-1.0-1.0-c0.2-is1@rs269ra-r3@train@scale=0.5,1.0,1.5,2.0
CAMS_DIR=$SCRATCH/PuzzleCAM/experiments/predictions/pnoc/coco14-rs269-pnoc-b16-a2-lr0.05-ls0-ow0.0-1.0-1.0-c0.2-is1@rs269ra-r1@train@scale=0.5,1.0,1.5,2.0
TAG=saliency/$DATASET-ccamh-$ARCH@rs269pnoc-lr0.05@rs269@b$BATCH_SIZE-fg$FG_T-lr$LR-b$BATCH_SIZE
# run_training
# run_inference

LR=0.001
MIXED_PRECISION=true
BATCH_SIZE=128
ACCUMULATE_STEPS=2
LABELSMOOTHING=0.1

LR=0.0005
FG_T=0.3
BATCH_SIZE=64
TAG=saliency/$DATASET-ccamh-$ARCH@rs269pnoc-lr0.05@rs269@b$BATCH_SIZE-fg$FG_T-lr$LR-b$BATCH_SIZE
# run_training
run_inference


###############
# Alternatives
###############

# BATCH_SIZE=32
# TAG=saliency/$DATASET-ccamh-$ARCH@rs269pnoc@rs269@b$BATCH_SIZE-fg$FG_T-lr$LR-b$BATCH_SIZE
# run_training

# BATCH_SIZE=64
# TAG=saliency/$DATASET-ccamh-$ARCH@rs269pnoc@rs269@b$BATCH_SIZE-fg$FG_T-lr$LR-b$BATCH_SIZE
# run_training

# FG_T=0.3
# TAG=saliency/$DATASET-ccamh-$ARCH@rs269pnoc@rs269@b$BATCH_SIZE-fg$FG_T-lr$LR-b$BATCH_SIZE
# run_training

# FG_T=0.2
# TAG=saliency/$DATASET-ccamh-$ARCH@rs269pnoc@rs269@b$BATCH_SIZE-fg$FG_T-lr$LR-b$BATCH_SIZE
# run_training
