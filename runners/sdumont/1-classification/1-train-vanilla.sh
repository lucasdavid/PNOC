#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH -p sequana_gpu_shared
#SBATCH -J tr-vanilla
#SBATCH -o /scratch/lerdl/lucas.david/logs/puzzle/van-%j.out
#SBATCH --time=48:00:00

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

echo "[puzzle/train.sequana] started running at $(date +'%Y-%m-%d %H:%M:%S')."

nodeset -e $SLURM_JOB_NODELIST

cd $SCRATCH/PuzzleCAM

module load sequana/current
module load gcc/7.4_sequana python/3.9.1_sequana cudnn/8.2_cuda-11.1_sequana


export PYTHONPATH=$(pwd)
# export OMP_NUM_THREADS=8

PY=python3.9
SOURCE=scripts/cam/train_vanilla.py
WORKERS=16

# DATASET=voc12
# DATA_DIR=$SCRATCH/datasets/VOCdevkit/VOC2012/
DATASET=coco14
DATA_DIR=$SCRATCH/datasets/coco14/

ARCH=rs269
ARCHITECTURE=resnest269
TRAINABLE_STEM=true
DILATED=false
MODE=normal
REGULAR=none

CUTMIX=0.5
MIXUP=1.
LABELSMOOTHING=0  # 0.1

IMAGE_SIZE=512
EPOCHS=15
BATCH=32


AUGMENT=colorjitter
TAG=$DATASET-$ARCH
CUDA_VISIBLE_DEVICES=0,1,2,3         \
$PY $SOURCE                          \
  --tag             $TAG             \
  --device          $DEVICE          \
  --num_workers     $WORKERS         \
  --batch_size      $BATCH           \
  --architecture    $ARCHITECTURE    \
  --dilated         $DILATED         \
  --mode            $MODE            \
  --trainable-stem  $TRAINABLE_STEM  \
  --regularization  $REGULAR         \
  --image_size      $IMAGE_SIZE      \
  --min_image_size  320              \
  --max_image_size  640              \
  --augment         $AUGMENT         \
  --cutmix_prob     $CUTMIX          \
  --mixup_prob      $MIXUP           \
  --label_smoothing $LABELSMOOTHING  \
  --max_epoch       $EPOCHS          \
  --dataset         $DATASET         \
  --data_dir        $DATA_DIR


# AUGMENT=colorjitter_randaugment
# TAG=$DATASET-$ARCH-ra
# CUTMIX=0.5
# CUDA_VISIBLE_DEVICES=0,1,2,3         \
#   --tag             $TAG             \
#   --device          $DEVICE          \
#   --num_workers     $WORKERS         \
#   --batch_size      $BATCH           \
#   --architecture    $ARCHITECTURE    \
#   --dilated         $DILATED         \
#   --mode            $MODE            \
#   --trainable-stem  $TRAINABLE_STEM  \
#   --regularization  $REGULAR         \
#   --image_size      $IMAGE_SIZE      \
#   --min_image_size  $IMAGE_SIZE      \
#   --max_image_size  $IMAGE_SIZE      \
#   --augment         $AUGMENT         \
#   --cutmix_prob     $CUTMIX          \
#   --mixup_prob      $MIXUP           \
#   --label_smoothing $LABELSMOOTHING  \
#   --max_epoch       $EPOCHS          \
#   --dataset         $DATASET         \
#   --data_dir        $DATA_DIR

# AUGMENT=colorjitter_randaugment_cutmix
# CUTMIX=0.5
# AUG=ra-cm$CUTMIX
# TAG=$DATASET-$ARCH-$AUG
# CUDA_VISIBLE_DEVICES=0,1,2,3         \
#   --tag             $TAG             \
#   --device          $DEVICE          \
#   --num_workers     $WORKERS         \
#   --batch_size      $BATCH           \
#   --architecture    $ARCHITECTURE    \
#   --dilated         $DILATED         \
#   --mode            $MODE            \
#   --trainable-stem  $TRAINABLE_STEM  \
#   --regularization  $REGULAR         \
#   --image_size      $IMAGE_SIZE      \
#   --min_image_size  $IMAGE_SIZE      \
#   --max_image_size  $IMAGE_SIZE      \
#   --augment         $AUGMENT         \
#   --cutmix_prob     $CUTMIX          \
#   --mixup_prob      $MIXUP           \
#   --label_smoothing $LABELSMOOTHING  \
#   --max_epoch       $EPOCHS          \
#   --dataset         $DATASET         \
#   --data_dir        $DATA_DIR


# AUGMENT=colorjitter_cutmix
# CUTMIX=0.5
# AUG=cm$CUTMIX
# TAG=$DATASET-$ARCH-$AUG
# CUDA_VISIBLE_DEVICES=0,1,2,3         \
#     $PY $SOURCE                      \
#   --tag             $TAG             \
#   --device          $DEVICE          \
#   --num_workers     $WORKERS         \
#   --batch_size      $BATCH           \
#   --architecture    $ARCHITECTURE    \
#   --dilated         $DILATED         \
#   --mode            $MODE            \
#   --trainable-stem  $TRAINABLE_STEM  \
#   --regularization  $REGULAR         \
#   --image_size      $IMAGE_SIZE      \
#   --min_image_size  $IMAGE_SIZE      \
#   --max_image_size  $IMAGE_SIZE      \
#   --augment         $AUGMENT         \
#   --cutmix_prob     $CUTMIX          \
#   --mixup_prob      $MIXUP           \
#   --label_smoothing $LABELSMOOTHING  \
#   --max_epoch       $EPOCHS          \
#   --dataset         $DATASET         \
#   --data_dir        $DATA_DIR
