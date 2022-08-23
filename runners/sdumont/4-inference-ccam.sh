#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH -p sequana_gpu_shared
#SBATCH -J mk-ccam
#SBATCH -o /scratch/lerdl/lucas.david/logs/ccam/mk-%j.out
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

PY=python3.9
SOURCE=ccam_inference.py

LOGS_DIR=$SCRATCH/logs/ccam
DATA_DIR=$SCRATCH/datasets/VOCdevkit/VOC2012/

WORKERS=8

DILATED=false
TRAINABLE_STEM=true
MODE=normal

DOMAIN=train_aug


TAG=ccamh@resnest101@resnest101-ra
ARCHITECTURE=resnest101
S4_OUT_FEATURES=1024
WEIGHTS=./experiments/models/ccamh@resnest101@resnest101-ra@0.3-@BG_T@h1.0-e10-b64-lr0.001.pth
# THRESHOLD=0.3
# CRF=10

CUDA_VISIBLE_DEVICES=0 $PY $SOURCE       \
  --tag             $TAG                 \
  --domain          $DOMAIN              \
  --num_workers     $WORKERS             \
  --architecture    $ARCHITECTURE        \
  --dilated         $DILATED             \
  --stage4_out_features $S4_OUT_FEATURES \
  --mode            $MODE                \
  --trainable-stem  $TRAINABLE_STEM      \
  --pretrained      $WEIGHTS             \
  --data_dir        $DATA_DIR            &


# TAG=ccamh@resnet38d@resnest101-ra@0.3-@BG_T@h1.0-e10-b64-lr0.001
# ARCHITECTURE=resnet38d
# WEIGHTS=./experiments/models/ccamh@resnet38d@resnest101-ra@0.3-@BG_T@h1.0-e10-b64-lr0.001.pth

# CUDA_VISIBLE_DEVICES=1 $PY $SOURCE       \
#   --tag             $TAG                 \
#   --domain          $DOMAIN              \
#   --num_workers     $WORKERS             \
#   --architecture    $ARCHITECTURE        \
#   --dilated         $DILATED             \
#   --stage4_out_features $S4_OUT_FEATURES \
#   --mode            $MODE                \
#   --trainable-stem  $TRAINABLE_STEM      \
#   --pretrained      $WEIGHTS             \
#   --data_dir        $DATA_DIR            &

TAG=ccamh@resnest101@resnest269-poc
ARCHITECTURE=resnest101
S4_OUT_FEATURES=1024
WEIGHTS=./experiments/models/ccamh@resnest101@resnest269-poc@0.4-@BG_T@h1.0-e10-b64-lr0.001.pth
# THRESHOLD=0.3
# CRF=10

CUDA_VISIBLE_DEVICES=2 $PY $SOURCE       \
  --tag             $TAG                 \
  --domain          $DOMAIN              \
  --num_workers     $WORKERS             \
  --architecture    $ARCHITECTURE        \
  --dilated         $DILATED             \
  --stage4_out_features $S4_OUT_FEATURES \
  --mode            $MODE                \
  --trainable-stem  $TRAINABLE_STEM      \
  --pretrained      $WEIGHTS             \
  --data_dir        $DATA_DIR            &


# TAG=ccamh@resnet38d@resnest269-poc
# ARCHITECTURE=resnet38d
# WEIGHTS=./experiments/models/ccamh@resnet38d@resnest269-poc@0.4-@BG_T@h1.0-e10-b64-lr0.001.pth

# CUDA_VISIBLE_DEVICES=3 $PY $SOURCE       \
#   --tag             $TAG                 \
#   --domain          $DOMAIN              \
#   --num_workers     $WORKERS             \
#   --architecture    $ARCHITECTURE        \
#   --dilated         $DILATED             \
#   --stage4_out_features $S4_OUT_FEATURES \
#   --mode            $MODE                \
#   --trainable-stem  $TRAINABLE_STEM      \
#   --pretrained      $WEIGHTS             \
#   --data_dir        $DATA_DIR            &

wait
