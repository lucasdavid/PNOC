#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH -p sequana_gpu_shared
#SBATCH -J tr-poc
#SBATCH -o /scratch/lerdl/lucas.david/logs/puzzle/poc-%j.out
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
SOURCE=train_classification_with_puzzle_oc.py
LOGS_DIR=$SCRATCH/logs/puzzle
DATA_DIR=$SCRATCH/datasets/VOCdevkit/VOC2012/

# Dataset
BATCH=16
# AUGMENT=colorjitter
# Arch
ARCHITECTURE=resnest269
REG=none
DILATED=false
TRAINABLE_STEM=true
# Training
EPOCHS=15
MODE=normal
# OC
OC_ARCHITECTURE=resnest269
OC_REG=none
OC_PRETRAINED=experiments/models/ResNeSt269.pth
OC_STRATEGY=random
OC_FOCAL_MOMENTUM=0.8
OC_FOCAL_GAMMA=5.0
# Schedule
P_INIT=0.0
P_ALPHA=4.0
P_SCHEDULE=0.5

OC_INIT=0.3
OC_ALPHA=1.0
OC_SCHEDULE=1.0

TAG=$ARCHITECTURE@$MODE@puzzleoc@b$BATCH-rep3

CUDA_VISIBLE_DEVICES=0,1,2,3               \
    $PY $SOURCE                            \
    --max_epoch         $EPOCHS            \
    --batch_size        $BATCH             \
    --architecture      $ARCHITECTURE      \
    --regularization    $REG               \
    --dilated           $DILATED           \
    --trainable-stem    $TRAINABLE_STEM    \
    --mode              $MODE              \
    --oc-architecture   $OC_ARCHITECTURE   \
    --oc-pretrained     $OC_PRETRAINED     \
    --oc-regularization $OC_REG            \
    --oc-strategy       $OC_STRATEGY       \
    --oc-alpha-schedule $OC_SCHEDULE       \
    --oc-focal-momentum $OC_FOCAL_MOMENTUM \
    --oc-focal-gamma    $OC_FOCAL_GAMMA    \
    --alpha             $P_ALPHA           \
    --alpha_init        $P_INIT            \
    --alpha_schedule    $P_SCHEDULE        \
    --oc-alpha          $OC_ALPHA          \
    --oc-alpha-init     $OC_INIT           \
    --tag               $TAG               \
    --data_dir          $DATA_DIR
    # --augment           $AUGMENT           \
