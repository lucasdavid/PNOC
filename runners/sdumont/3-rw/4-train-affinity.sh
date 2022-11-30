#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH -p sequana_gpu_shared
#SBATCH -J tr-aff
#SBATCH -o /scratch/lerdl/lucas.david/logs/puzzle/affinitynet/train-%j.out
#SBATCH --time=8:00:00

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

echo "[sdumont/sequana/classification/train-puzzle] started running at $(date +'%Y-%m-%d %H:%M:%S')."

nodeset -e $SLURM_JOB_NODELIST

cd $SCRATCH/PuzzleCAM

module load sequana/current
module load gcc/7.4_sequana python/3.9.1_sequana cudnn/8.2_cuda-11.1_sequana

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$(pwd)

PY=python3.9
SOURCE=scripts/rw/train.py
DATA_DIR=$SCRATCH/datasets/VOCdevkit/VOC2012/

# Architecture
ARCHITECTURE=resnest269
BATCH_SIZE=32
LR=0.1

LABEL=affnet@rs269-poc@pn-fgh@crf-10-gt-0.9@aff_fg=0.40_bg=0.10
TAG=affnet@rs269-poc@pn-fgh@crf-10-gt-0.9@aff_fg=0.40_bg=0.10
CAM_DIR=ResNeSt269@PuzzleOc@train@scale=0.5,1.0,1.5,2.0

# CUDA_VISIBLE_DEVICES=0,1,2,3     \
# $PY $SOURCE                      \
#     --architecture $ARCHITECTURE \
#     --tag          $TAG          \
#     --label_name   $LABEL        \
#     --batch_size   $BATCH_SIZE   \
#     --lr           $LR           \
#     --data_dir     $DATA_DIR

CUDA_VISIBLE_DEVICES=0           \
$PY inference_rw.py              \
    --architecture $ARCHITECTURE \
    --model_name   $TAG          \
    --cam_dir      $CAM_DIR      \
    --domain       train_aug     \
    --beta         10            \
    --exp_times    8             \
    --data_dir     $DATA_DIR

RW_DIR=$TAG@train@beta=10@exp_times=8@rw

$PY evaluate.py                         \
  --experiment_name "$RW_DIR"           \
  --domain train                        \
  --gt_dir "$DATA_DIR"SegmentationClass \
  --min_th 0.05                         \
  --max_th 0.9                          \
  --step_th 0.05
