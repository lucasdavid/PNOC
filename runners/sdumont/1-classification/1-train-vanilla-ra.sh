#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH -p sequana_gpu_shared
#SBATCH -J tr-vanilla
#SBATCH -o /scratch/lerdl/lucas.david/logs/puzzle/van-%j.out
#SBATCH --time=16:00:00

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
# task over the VOC12 or COCO14 dataset.
#

echo "[sdumont/sequana/classification/train-vanilla] started running at $(date +'%Y-%m-%d %H:%M:%S')."

nodeset -e $SLURM_JOB_NODELIST

cd $SCRATCH/PuzzleCAM

module load sequana/current
module load gcc/7.4_sequana python/3.9.1_sequana cudnn/8.2_cuda-11.1_sequana

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$(pwd)

PY=python3.9
SOURCE=scripts/cam/train_vanilla.py

# DATASET=voc12
# DATA_DIR=$SCRATCH/datasets/VOCdevkit/VOC2012/
DATASET=coco14
DATA_DIR=$SCRATCH/datasets/coco14/

ARCH=rs101
ARCHITECTURE=resnest101
REGULAR=none

AUGMENT=colorjitter_randaugment
CUTMIX_PROB=0.5
AUG=ra

TAG=$DATASET-$ARCH-$AUG

CUDA_VISIBLE_DEVICES=0,1,2,3         \
    $PY $SOURCE                      \
    --architecture   $ARCHITECTURE   \
    --regularization $REGULAR        \
    --augment        $AUGMENT        \
    --cutmix_prob    $CUTMIX_PROB    \
    --tag            $TAG            \
    --dataset        $DATASET        \
    --data_dir       $DATA_DIR
