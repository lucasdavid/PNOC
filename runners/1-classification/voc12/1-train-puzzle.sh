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
# Train ResNeSt269 to perform multilabel classification task
# over the VOC12 or COCO14 dataset using Puzzle-CAM strategy.
#

echo "[sdumont/sequana/classification/train-puzzle] started running at $(date +'%Y-%m-%d %H:%M:%S')."

nodeset -e $SLURM_JOB_NODELIST

cd $SCRATCH/PuzzleCAM

module load sequana/current
module load gcc/7.4_sequana python/3.9.1_sequana cudnn/8.2_cuda-11.1_sequana

export PYTHONPATH=$(pwd)
# export OMP_NUM_THREADS=16

PY=python3.9
SOURCE=scripts/cam/train_puzzle.py
WORKERS=16
DEVICES=0,1,2,3

DATASET=voc12
DATA_DIR=$SCRATCH/datasets/VOCdevkit/VOC2012/
DOMAIN=train
# DATASET=coco14
# DATA_DIR=$SCRATCH/datasets/coco14/
# DOMAIN=train2014


run_experiment() {
  echo "Running $TAG experiment"
  CUDA_VISIBLE_DEVICES=$DEVICES        \
  $PY $SOURCE                          \
    --tag             $TAG             \
    --num_workers     $WORKERS         \
    --batch_size      $BATCH           \
    --architecture    $ARCHITECTURE    \
    --dilated         $DILATED         \
    --mode            $MODE            \
    --trainable-stem  $TRAINABLE_STEM  \
    --regularization  $REGULAR         \
    --re_loss_option  masking          \
    --re_loss         L1_Loss          \
    --alpha_schedule  0.50             \
    --alpha           4.00             \
    --image_size      $IMAGE_SIZE      \
    --min_image_size  $MIN_IMAGE_SIZE  \
    --max_image_size  $MAX_IMAGE_SIZE  \
    --augment         $AUGMENT         \
    --cutmix_prob     $CUTMIX          \
    --mixup_prob      $MIXUP           \
    --label_smoothing $LABELSMOOTHING  \
    --max_epoch       $EPOCHS          \
    --dataset         $DATASET         \
    --data_dir        $DATA_DIR
}

run_inference () {
    CUDA_VISIBLE_DEVICES=$DEVICES        \
    $PY scripts/cam/inference.py         \
    --architecture      $ARCHITECTURE    \
    --regularization    $REGULAR         \
    --dilated           $DILATED         \
    --trainable-stem    $TRAINABLE_STEM  \
    --mode              $MODE            \
    --tag               $TAG             \
    --domain            $DOMAIN          \
    --data_dir          $DATA_DIR
}

IMAGE_SIZE=512
MIN_IMAGE_SIZE=320
MAX_IMAGE_SIZE=640
EPOCHS=15
BATCH=32

ARCH=rs101
ARCHITECTURE=resnest101
TRAINABLE_STEM=true
DILATED=false
MODE=normal
REGULAR=none

CUTMIX=0.5        # requires AUGMENT=cutmix[...]
MIXUP=1.          # requires AUGMENT=mixup[...]
LABELSMOOTHING=0  # default=0.1

ARCH=rs101
ARCHITECTURE=resnest101
AUGMENT=colorjitter
# ARCH=rs269
# ARCHITECTURE=resnest269
# AUGMENT=colorjitter
TAG=$DATASET-$ARCH-p-r3
run_experiment
run_inference

# ARCH=rs101
# ARCHITECTURE=resnest101
# BATCH=32
# LABELSMOOTHING=0.1
# CUTMIX=1.0
# AUGMENT=randaugment_cutormixup
# TAG=$DATASET-$ARCH-p-ls$LABELSMOOTHING-ra-cutormixup
# run_experiment

# ARCH=rs101
# ARCHITECTURE=resnest101
# BATCH=32
# LABELSMOOTHING=0.1
# MIXUP=1.0
# AUGMENT=randaugment_mixup
# MIN_IMAGE_SIZE=512
# MAX_IMAGE_SIZE=512
# TAG=$DATASET-$ARCH-p-ls$LABELSMOOTHING-ra-mixup
# run_experiment