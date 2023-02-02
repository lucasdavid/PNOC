#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH -p sequana_gpu_shared
#SBATCH -J inf-aff
#SBATCH -o /scratch/lerdl/lucas.david/logs/puzzle/affinitynet/inf-%j.out
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


export PYTHONPATH=$(pwd)

PY=python3.9
SOURCE=scripts/rw/inference.py
WORKERS=8
DEVICES=0,1,2,3

# Dataset
IMAGE_SIZE=512
MIN_IMAGE_SIZE=320
MAX_IMAGE_SIZE=640

run() {
    CUDA_VISIBLE_DEVICES=$DEVICES    \
    $PY $SOURCE                      \
        --architecture $ARCHITECTURE \
        --image_size   $IMAGE_SIZE   \
        --model_name   $MODEL_NAME   \
        --cam_dir      $CAM_DIR      \
        --domain       $DOMAIN       \
        --beta         10            \
        --exp_times    8             \
        --data_dir     $DATA_DIR
}


DATASET=voc12
DATA_DIR=$SCRATCH/datasets/VOCdevkit/VOC2012/
DOMAIN=train

ARCHITECTURE=resnest269
# MODEL_NAME=affnet@rs269-poc@pn-fgh@crf-10-gt-0.9@aff_fg=0.40_bg=0.10
# CAM_DIR=poc/ResNeSt269@PuzzleOc@train@scale=0.5,1.0,1.5,2.0
# RW_DIR=rw/$MODEL_NAME@train@beta=10@exp_times=8@rw
# run

DOMAIN=val

MODEL_NAME=rw/AffinityNet@ResNeSt-269@Puzzle
CAM_DIR=puzzle/ResNeSt269@Puzzle@optimal@$DOMAIN@scale=0.5,1.0,1.5,2.0
RW_DIR=$MODEL_NAME@$DOMAIN@beta=10@exp_times=8@rw
run

MODEL_NAME=rw/AffinityNet@resnest269@puzzlerep@aff_fg=0.40_bg=0.10
CAM_DIR=puzzle/resnest269@puzzlerep@$DOMAIN@scale=0.5,1.0,1.5,2.0
RW_DIR=$MODEL_NAME@$DOMAIN@beta=10@exp_times=8@rw
run

MODEL_NAME=rw/AffinityNet@ResNeSt269@PuzzleOc
CAM_DIR=poc/ResNeSt269@PuzzleOc@$DOMAIN@scale=0.5,1.0,1.5,2.0
RW_DIR=$MODEL_NAME@$DOMAIN@beta=10@exp_times=8@rw
run

MODEL_NAME=rw/voc12-an@rs269poc-ls0.1@fg0.4-bg0.1-crf10-gt0.9@aff_fg=0.40_bg=0.10
CAM_DIR=poc/voc12-rs269-poc-ls0.1@rs269ra-r3@$DOMAIN@scale=0.5,1.0,1.5,2.0
RW_DIR=$MODEL_NAME@$DOMAIN@beta=10@exp_times=8@rw
run

MODEL_NAME=rw/voc12-an@ccamh@rs269poc-ls0.1@fg0.4-bg0.1-crf10-gt0.9@aff_fg=0.40_bg=0.10
CAM_DIR=poc/voc12-rs269-poc-ls0.1@rs269ra-r3@$DOMAIN@scale=0.5,1.0,1.5,2.0
RW_DIR=$MODEL_NAME@$DOMAIN@beta=10@exp_times=8@rw
run

MODEL_NAME=rw/voc12-an@rs269apoc-ls0.1@fg0.3-bg0.1-crf10-gt0.7@aff_fg=0.30_bg=0.10
CAM_DIR=apoc/voc12-rs269-apoc-ls0.1-ow0.0-1.0-1.0-cams-0.2-octis1-amp@rs269ra-r3@$DOMAIN@scale=0.5,1.0,1.5,2.0
RW_DIR=$MODEL_NAME@$DOMAIN@beta=10@exp_times=8@rw
run

# MODEL_NAME=rw/voc12-an@rs269apoc-ls0.1@fg0.3-bg0.1-crf10-gt0.7@aff_fg=0.30_bg=0.10

## ===================================
## COCO 14

# DATASET=coco14
# DATA_DIR=/home/ldavid/workspace/datasets/coco14/
# DOMAIN=train2014
# MODEL_NAME=
# CAM_DIR=
# RW_DIR=
# run
