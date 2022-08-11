#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH -p sequana_gpu_shared
#SBATCH -J tr-puzzle
#SBATCH -o /scratch/lerdl/lucas.david/logs/puzzle/puz-%j.out
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
# task over Pascal VOC 2012 using OC-CSE strategy.
#

echo "[voc12/puzzle/train.sequana] started running at $(date +'%Y-%m-%d %H:%M:%S')."

nodeset -e $SLURM_JOB_NODELIST

cd $SCRATCH/PuzzleCAM

module load sequana/current
module load gcc/7.4_sequana python/3.9.1_sequana cudnn/8.2_cuda-11.1_sequana

PY=python3.9
SOURCE=train_classification_with_puzzle.py
DATA_DIR=$SCRATCH/datasets/VOCdevkit/VOC2012/

ARCHITECTURE=resnest269
BATCH=16
TAG=$ARCHITECTURE@puzzlerep2

CUDA_VISIBLE_DEVICES=0,1,2,3        \
    $PY $SOURCE                     \
    --architecture   $ARCHITECTURE  \
    --batch_size     $BATCH         \
    --mode           normal         \
    --re_loss_option masking        \
    --re_loss        L1_Loss        \
    --alpha_schedule 0.50           \
    --alpha          4.00           \
    --tag            $TAG           \
    --data_dir       $DATA_DIR
