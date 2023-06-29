#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH -p sequana_gpu_shared
#SBATCH -J inf
#SBATCH -o /scratch/lerdl/lucas.david/logs/puzzle/inf-%j.out
#SBATCH --time=1:00:00

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
# Infer CAMs using a multi-label classification model previous
# trained trained over the VOC12 or COCO14 dataset.
# TTA is employed to produce higher quality maps.
#

echo "[sdumont/sequana/classification/inference] started running at $(date +'%Y-%m-%d %H:%M:%S')."

nodeset -e $SLURM_JOB_NODELIST

cd $SCRATCH/PuzzleCAM

module load sequana/current
module load gcc/7.4_sequana python/3.9.1_sequana cudnn/8.2_cuda-11.1_sequana
# module load gcc/7.4 python/3.9.1 cudnn/8.2_cuda-11.1


export PYTHONPATH=$(pwd)

PY=python3.9
SOURCE=scripts/cam/inference.py
DEVICES=0,1,2,3

DATASET=voc12
DATA_DIR=$SCRATCH/datasets/VOCdevkit/VOC2012/
DOMAIN=train
# DATASET=coco14
# DATA_DIR=$SCRATCH/datasets/coco14/
# DOMAIN=train2014

ARCHITECTURE=resnest269
DILATED=false
MODE=normal
REG=none

run_inference () {
    CUDA_VISIBLE_DEVICES=$DEVICES           \
    $PY $SOURCE                             \
    --architecture   $ARCHITECTURE          \
    --dilated        $DILATED               \
    --regularization $REG                   \
    --mode           $MODE                  \
    --weights        $WEIGHTS               \
    --tag            $TAG                   \
    --domain         $DOMAIN                \
    --dataset        $DATASET               \
    --data_dir       $DATA_DIR
}

# ARCHITECTURE=resnest269
# WEIGHTS=voc12-rs269-poc-ls0.1@rs269ra-r3
# TAG=poc/$WEIGHTS
# DOMAIN=train_aug
# run_inference
# DOMAIN=val
# run_inference

## RA
#
# ARCHITECTURE=resnest269
# WEIGHTS=cam/resnest269@randaug
# TAG=vanilla/rs269ra
# run_inference


## A-P-OC
#
# ARCHITECTURE=resnest269
# WEIGHTS=poc/voc12-rs269-poc-ls0.1@rs269ra-r3
# TAG=$WEIGHTS
# DOMAIN=train_aug
# run_inference
# DOMAIN=val
# run_inference

DOMAIN=val
WEIGHTS=puzzle/ResNeSt269@Puzzle@optimal
TAG=$WEIGHTS
run_inference

WEIGHTS=puzzle/resnest269@puzzlerep
TAG=$WEIGHTS
run_inference

WEIGHTS=puzzle/resnest269@puzzlerep2
TAG=$WEIGHTS
run_inference

WEIGHTS=poc/ResNeSt269@PuzzleOc
TAG=$WEIGHTS
run_inference

# WEIGHTS=poc/voc12-rs269-poc-ls0.1@rs269ra-r3
# run_inference
# WEIGHTS=pnoc/voc12-rs269-pnoc-ls0.1-ow0.0-1.0-1.0-cams-0.2-octis1-amp@rs269ra-r3
# run_inference