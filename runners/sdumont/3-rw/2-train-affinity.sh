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


export PYTHONPATH=$(pwd)

PY=python3.9
SOURCE=scripts/rw/train_affinity.py
WORKERS=8
DEVICES=0,1,2,3

# Dataset
IMAGE_SIZE=512
MIN_IMAGE_SIZE=320
MAX_IMAGE_SIZE=640

# Architecture
ARCHITECTURE=resnest269
ARCH=rs269
BATCH_SIZE=32
LR=0.1


run_training() {
    echo "============================================================"
    echo "Experiment $TAG"
    echo "============================================================"

    CUDA_VISIBLE_DEVICES=$DEVICES    \
    WANDB_TAGS="$DATASET,$ARCH,rw"   \
    $PY $SOURCE                      \
        --architecture $ARCHITECTURE \
        --tag          $TAG       \
        --batch_size   $BATCH_SIZE   \
        --image_size   $IMAGE_SIZE   \
        --min_image_size $MIN_IMAGE_SIZE \
        --max_image_size $MAX_IMAGE_SIZE \
        --dataset      $DATASET      \
        --lr           $LR           \
        --label_dir    $LABEL_DIR    \
        --data_dir     $DATA_DIR     \
        --num_workers  $WORKERS
}

run_inference() {
  CUDA_VISIBLE_DEVICES=$DEVICES    \
  $PY scripts/rw/inference.py      \
      --architecture $ARCHITECTURE \
      --model_name $TAG            \
      --cam_dir $CAMS_DIR          \
      --beta 10                    \
      --exp_times 8                \
      --dataset $DATASET           \
      --domain $DOMAIN             \
      --data_dir $DATA_DIR
    
  # CUDA_VISIBLE_DEVICES=1           \
  # $PY scripts/rw/inference.py      \
  #     --architecture $ARCHITECTURE \
  #     --model_name $TAG            \
  #     --cam_dir $CAMS_DIR          \
  #     --beta 10                    \
  #     --exp_times 8                \
  #     --dataset $DATASET           \
  #     --domain val                 \
  #     --data_dir $DATA_DIR         &
}


DATASET=voc12
DATA_DIR=$SCRATCH/datasets/VOCdevkit/VOC2012/
DOMAIN=train
# DATASET=coco14
# DATA_DIR=$SCRATCH/datasets/coco14/
# DOMAIN=train2014

# TAG=rw/voc12-an@rs269poc-ls0.1@fg0.4-bg0.1-crf10-gt0.9@aff_fg=0.40_bg=0.10
# CAMS_DIR=./experiments/predictions/poc/voc12-rs269-poc-ls0.1@rs269ra-r3@train@scale=0.5,1.0,1.5,2.0
# LABEL_DIR=./experiments/predictions/voc12-an@rs269poc-ls0.1@fg0.4-bg0.1-crf10-gt0.9@aff_fg=0.40_bg=0.10
# run_training
# run_inference

# TAG=rw/voc12-an@rs269apoc-ls0.1@fg0.3-bg0.1-crf10-gt0.9@aff_fg=0.30_bg=0.10
# CAMS_DIR=./experiments/predictions/apoc/voc12-rs269-apoc-ls0.1-ow0.0-1.0-1.0-cams-0.2-octis1-amp@rs269ra-r3@train@scale=0.5,1.0,1.5,2.0
# LABEL_DIR=./experiments/predictions/voc12-an@rs269apoc-ls0.1@fg0.3-bg0.1-crf10-gt0.9@aff_fg=0.30_bg=0.10
# run_training
# run_inference

# TAG=rw/voc12-an@ccamh@rs269poc-ls0.1@fg0.4-bg0.1-crf10-gt0.9@aff_fg=0.40_bg=0.10
# CAMS_DIR=./experiments/predictions/poc/voc12-rs269-poc-ls0.1@rs269ra-r3@train@scale=0.5,1.0,1.5,2.0
# LABEL_DIR=./experiments/predictions/voc12-an@ccamh@rs269poc-ls0.1@fg0.4-bg0.1-crf10-gt0.9@aff_fg=0.40_bg=0.10
# run_training
# run_inference

# TAG=rw/voc12-an@ccamh@rs269apoc-ls0.1@fg0.3-bg0.1-crf10-gt0.9@aff_fg=0.30_bg=0.10
# CAMS_DIR=./experiments/predictions/apoc/voc12-rs269-apoc-ls0.1-ow0.0-1.0-1.0-cams-0.2-octis1-amp@rs269ra-r3@train@scale=0.5,1.0,1.5,2.0
# LABEL_DIR=./experiments/predictions/voc12-an@ccamh@rs269apoc-ls0.1@fg0.3-bg0.1-crf10-gt0.9@aff_fg=0.30_bg=0.10
# run_training
# run_inference

# TAG=rw/voc12-an@rs269apoc-ls0.1@fg0.3-bg0.1-crf10-gt0.7@aff_fg=0.30_bg=0.10
# CAMS_DIR=./experiments/predictions/apoc/voc12-rs269-apoc-ls0.1-ow0.0-1.0-1.0-cams-0.2-octis1-amp@rs269ra-r3@train@scale=0.5,1.0,1.5,2.0
# LABEL_DIR=./experiments/predictions/voc12-an@rs269apoc-ls0.1@fg0.3-bg0.1-crf10-gt0.7@aff_fg=0.30_bg=0.10
# run_training
# run_inference

# TAG=rw/voc12-an@rs269apoc-ls0.1@fg0.3-bg0.05-crf10-gt0.7@aff_fg=0.30_bg=0.05
# CAMS_DIR=./experiments/predictions/apoc/voc12-rs269-apoc-ls0.1-ow0.0-1.0-1.0-cams-0.2-octis1-amp@rs269ra-r3@train@scale=0.5,1.0,1.5,2.0
# LABEL_DIR=./experiments/predictions/voc12-an@rs269apoc-ls0.1@fg0.3-bg0.05-crf10-gt0.7@aff_fg=0.30_bg=0.05
# run_training
# run_inference

TAG=rw/voc12-an@ccamh@rs269apoc-ls0.1@fg0.3-bg0.1-crf10-gt0.7
# CAMS_DIR=./experiments/predictions/apoc/voc12-rs269-apoc-ls0.1-ow0.0-1.0-1.0-cams-0.2-octis1-amp@rs269ra-r3@train@scale=0.5,1.0,1.5,2.0
# LABEL_DIR=./experiments/predictions/voc12-an@ccamh@rs269apoc-ls0.1@crf10-gt0.7@aff_fg=0.30_bg=0.10
# run_training
CAMS_DIR=./experiments/predictions/apoc/voc12-rs269-apoc-ls0.1-ow0.0-1.0-1.0-cams-0.2-octis1-amp@rs269ra-r3@train@scale=0.5,1.0,1.5,2.0
DOMAIN=train_aug
run_inference
# run_inference
# CAMS_DIR=./experiments/predictions/apoc/voc12-rs269-apoc-ls0.1-ow0.0-1.0-1.0-cams-0.2-octis1-amp@rs269ra-r3@val@scale=0.5,1.0,1.5,2.0
# DOMAIN=val
# run_inference

# DATASET=coco14
# DATA_DIR=/home/ldavid/workspace/datasets/coco14/
# DOMAIN=train2014
# TAG=rw/coco14-an@...
# LABEL_DIR=
# run_training
