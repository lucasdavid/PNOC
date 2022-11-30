#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH -p sequana_gpu_shared
#SBATCH -J tr-ccam
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

cd $SCRATCH/PuzzleCAM

module load sequana/current
module load gcc/7.4_sequana python/3.9.1_sequana cudnn/8.2_cuda-11.1_sequana
# module load gcc/7.4 python/3.9.1 cudnn/8.2_cuda-11.1

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$(pwd)

PY=python3.9
SOURCE=scripts/ccam/train_with_cam_hints.py

LOGS_DIR=$SCRATCH/logs/ccam
DATA_DIR=$SCRATCH/datasets/VOCdevkit/VOC2012/
# CAMS_DIR=$SCRATCH/PuzzleCAM/experiments/predictions/resnest101@randaug@train@scale=0.5,1.0,1.5,2.0
CAMS_DIR=$SCRATCH/PuzzleCAM/experiments/predictions/ResNeSt269@PuzzleOc@train@scale=0.5,1.0,1.5,2.0

WORKERS=8

IMAGE_SIZE=448
EPOCHS=10
BATCH_SIZE=32
ACCUMULATE_STEPS=4

ARCHITECTURE=resnest269
DILATED=false
TRAINABLE_STEM=true
MODE=normal
S4_OUT_FEATURES=1024

ALPHA=0.25
HINT_W=1.0
LR=0.001

FG_T=0.4
# BG_T=0.1

TAG=ccam-fg-hints@$ARCHITECTURE@rs269-poc@$FG_T@h$HINT_W-e$EPOCHS-b$BATCH_SIZE-lr$LR

CUDA_VISIBLE_DEVICES=0,1,2,3 $PY $SOURCE \
  --tag             $TAG                 \
  --alpha           $ALPHA               \
  --hint_w          $HINT_W              \
  --max_epoch       $EPOCHS              \
  --batch_size      $BATCH_SIZE          \
  --lr              $LR                  \
  --accumule_steps  $ACCUMULATE_STEPS    \
  --num_workers     $WORKERS             \
  --architecture    $ARCHITECTURE        \
  --stage4_out_features $S4_OUT_FEATURES \
  --dilated         $DILATED             \
  --mode            $MODE                \
  --trainable-stem  $TRAINABLE_STEM      \
  --image_size      $IMAGE_SIZE          \
  --cams_dir        $CAMS_DIR            \
  --fg_threshold    $FG_T                \
  --data_dir        $DATA_DIR
  # --bg_threshold    $BG_T                \
