#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH -p sequana_gpu_shared
#SBATCH -J rw-aff
#SBATCH -o /scratch/lerdl/lucas.david/logs/puzzle/affinitynet/rw-%j.out
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
# CAMs Inference.
#

echo "[voc12/puzzle/train.sequana] started running at $(date +'%Y-%m-%d %H:%M:%S')."

nodeset -e $SLURM_JOB_NODELIST

cd $SCRATCH/PuzzleCAM

# module load sequana/current
# module load gcc/7.4_sequana python/3.9.1_sequana cudnn/8.2_cuda-11.1_sequana
module load gcc/7.4 python/3.9.1 cudnn/8.2_cuda-11.1

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$(pwd)

PY=python3.9
SOURCE=inference_rw.py
DATA_DIR=$SCRATCH/datasets/VOCdevkit/VOC2012/

# Architecture
ARCHITECTURE=resnest269
# Dataset
TAG=AffinityNet@resnest269@puzzlerep@aff_fg=0.40_bg=0.10
CAM_DIR=resnest269@puzzlerep@train@scale=0.5,1.0,1.5,2.0
RW_DIR=$TAG@train@beta=10@exp_times=8@rw

CUDA_VISIBLE_DEVICES=0           \
$PY $SOURCE                      \
    --architecture $ARCHITECTURE \
    --model_name   $TAG          \
    --cam_dir      $CAM_DIR      \
    --domain       train_aug     \
    --beta         10            \
    --exp_times    8             \
    --data_dir     $DATA_DIR     &

# Parallel:
# PRED_DIR=./experiments/predictions/$RW_DIR@crf=1

# $PY make_pseudo_labels.py        \
#     --experiment_name $RW_DIR    \
#     --domain          train_aug  \
#     --threshold       0.35       \
#     --crf_iteration   1          \
#     --pred_dir        $PRED_DIR  \
#     --data_dir        $DATA_DIR  &

# SAL_DIR=$SCRATCH/logs/sal_55_epoch9_moco
# PRED_DIR=./experiments/predictions/$RW_DIR@crf=1@sal-moco

# $PY make_pseudo_labels.py        \
#     --experiment_name $RW_DIR    \
#     --domain          train_aug  \
#     --threshold       0.35       \
#     --crf_iteration   1          \
#     --sal_dir         $SAL_DIR   \
#     --pred_dir        $PRED_DIR  \
#     --data_dir        $DATA_DIR  &

wait
