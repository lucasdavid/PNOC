#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH -p sequana_gpu_shared
#SBATCH -J poc-rs101-rand
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

ARCHITECTURE=resnet38d
DILATED=true
OC_ARCHITECTURE=mcar
OC_PRETRAINED=/scratch/lerdl/lucas.david/MCAR/mcar-work-dirs/voc2012-resnet101-avg-512-4-0.5/model_best_95.1325.pth.tar
EPOCHS=15
BATCH_SIZE=16
OC_INIT=0.3
OC_ALPHA=1.0
OC_STRATEGY=random
OC_FOCAL_GAMMA=5.0
MODE=normal
REG=none
TAG=$ARCHITECTURE@mcar

CUDA_VISIBLE_DEVICES=0,1,2,3            \
    $PY $SOURCE                         \
    --max_epoch $EPOCHS                 \
    --batch_size $BATCH_SIZE            \
    --architecture $ARCHITECTURE        \
    --regularization $REG               \
    --mode $MODE                        \
    --alpha 4.00                        \
    --alpha_schedule 0.5                \
    --oc-architecture $OC_ARCHITECTURE  \
    --oc-pretrained $OC_PRETRAINED      \
    --oc-strategy $OC_STRATEGY          \
    --oc-alpha $OC_ALPHA                \
    --oc-alpha-init $OC_INIT            \
    --oc-alpha-schedule 1.0             \
    --oc-focal-gamma $OC_FOCAL_GAMMA    \
    --tag $TAG                          \
    --dilated $DILATED                  \
    --data_dir $DATA_DIR
