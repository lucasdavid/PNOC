#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH -p sequana_gpu_shared
#SBATCH -J tr-ccam
#SBATCH -o /scratch/lerdl/lucas.david/logs/ccam/tr-%j.out
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
# Train ResNeSt269 to perform multilabel classification
# task over Pascal VOC 2012 using OC-CSE strategy.
#

echo "[voc12/puzzle/train.sequana] started running at $(date +'%Y-%m-%d %H:%M:%S')."

nodeset -e $SLURM_JOB_NODELIST

cd $SCRATCH/PuzzleCAM

module load sequana/current
module load gcc/7.4_sequana python/3.9.1_sequana cudnn/8.2_cuda-11.1_sequana
# module load gcc/7.4 python/3.9.1 cudnn/8.2_cuda-11.1

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$(pwd)

PY=python3.9
SOURCE=ccam_train.py

LOGS_DIR=$SCRATCH/logs/ccam
DATA_DIR=$SCRATCH/datasets/VOCdevkit/VOC2012/

WORKERS=8

IMAGE_SIZE=448
EPOCHS=10
BATCH_SIZE=64
ACCUMULATE_STEPS=1

ARCHITECTURE=resnet50
ARCH=rn50
DILATED=false
TRAINABLE_STEM=true
MODE=normal
WEIGHTS=./experiments/models/moco_r50_v2-e3b0c442.pth  # imagenet
S4_OUT_FEATURES=1024

ALPHA=0.25
LR=0.0001

TAG=ccam-$ARCH-moco-e$EPOCHS-b$BATCH_SIZE-lr$LR-r2

CUDA_VISIBLE_DEVICES=0,1,2,3 $PY $SOURCE \
  --tag             $TAG                 \
  --alpha           $ALPHA               \
  --max_epoch       $EPOCHS              \
  --batch_size      $BATCH_SIZE          \
  --lr              $LR                  \
  --accumule_steps  $ACCUMULATE_STEPS    \
  --num_workers     $WORKERS             \
  --architecture    $ARCHITECTURE        \
  --stage4_out_features $S4_OUT_FEATURES \
  --dilated         $DILATED             \
  --mode            $MODE                \
  --weights         $WEIGHTS             \
  --trainable-stem  $TRAINABLE_STEM      \
  --image_size      $IMAGE_SIZE          \
  --data_dir        $DATA_DIR
