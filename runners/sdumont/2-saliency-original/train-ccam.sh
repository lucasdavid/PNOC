#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH -p sequana_gpu_shared
#SBATCH -J tr-ccam
#SBATCH -o /scratch/lerdl/lucas.david/logs/ccam/tr-baseline-%j.out
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

cd $SCRATCH/PuzzleCAM/CCAM/WSSS/

module load sequana/current
module load gcc/7.4_sequana python/3.9.1_sequana cudnn/8.2_cuda-11.1_sequana
# module load gcc/7.4 python/3.9.1 cudnn/8.2_cuda-11.1

PY=python3.9
SOURCE=train_CCAM_VOC12.py

LOGS_DIR=$SCRATCH/logs/ccam
DATA_DIR=$SCRATCH/datasets/VOCdevkit/VOC2012/

ARCH=resnest101
PRE=supervised
LR=0.0001
BT=64
TAG=rs101-lr$LR

OMP_NUM_THREADS=16       \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
$PY $SOURCE              \
  --architecture $ARCH   \
  --pretrained   $PRE    \
  --tag          $TAG    \
  --batch_size $BT       \
  --lr         $LR       \
  --alpha      0.25      \
  --data_dir   $DATA_DIR

OMP_NUM_THREADS=16       \
CUDA_VISIBLE_DEVICES=0   \
$PY inference_CCAM.py    \
  --architecture $ARCH   \
  --pretrained   $PRE    \
  --tag          $TAG    \
  --domain train         \
  --data_dir $DATA_DIR
