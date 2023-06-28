#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH -p sequana_gpu_shared
#SBATCH -J tr-poc
#SBATCH -o /scratch/lerdl/lucas.david/logs/puzzle/poc-%j.out
#SBATCH --time=96:00:00

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
# Train ResNeSt269 to perform multilabel classification task over
# the VOC12 or COCO14 dataset using Puzzle and OC-CSE strategies.
#

echo "[sdumont/sequana/classification/train-poc] started running at $(date +'%Y-%m-%d %H:%M:%S')."

nodeset -e $SLURM_JOB_NODELIST

cd $SCRATCH/PuzzleCAM

module load sequana/current
module load gcc/7.4_sequana python/3.9.1_sequana cudnn/8.2_cuda-11.1_sequana
# module load gcc/7.4 python/3.9.1 cudnn/8.2_cuda-11.1

# export LD_LIBRARY_PATH=$SCRATCH/.local/lib/python3.9/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$(pwd)
# export OMP_NUM_THREADS=16

PY=python3.9
SOURCE=scripts/cam/train_puzzle.py
WORKERS=32
DEVICES=0,1,2,3

DATASET=coco14
DATA_DIR=$SCRATCH/datasets/coco14/
DOMAIN=train2014


ARCH=rs101
ARCHITECTURE=resnest101
TRAINABLE_STEM=true
DILATED=false
MODE=normal
REGULAR=none

IMAGE_SIZE=512
MIN_IMAGE_SIZE=320
MAX_IMAGE_SIZE=640
EPOCHS=15
BATCH=32
LR=0.1
ACCUMULATE_STEPS=1

# Schedule
P_INIT=0.0
P_ALPHA=4.0
P_SCHEDULE=0.5

AUGMENT=colorjitter
CUTMIX=0.5
MIXUP=1.
LABELSMOOTHING=0  # 0.1


run_training() {
  echo "Running $TAG experiment"
  CUDA_VISIBLE_DEVICES=$DEVICES        \
  WANDB_RUN_GROUP="$W_GROUP"           \
  WANDB_TAGS="$W_TAGS"                 \
  $PY $SOURCE                          \
    --tag             $TAG             \
    --num_workers     $WORKERS         \
    --batch_size      $BATCH           \
    --accumulate_steps $ACCUMULATE_STEPS \
    --lr              $LR              \
    --architecture    $ARCHITECTURE    \
    --dilated         $DILATED         \
    --mode            $MODE            \
    --trainable-stem  $TRAINABLE_STEM  \
    --regularization  $REGULAR         \
    --re_loss_option  masking          \
    --re_loss         L1_Loss          \
    --alpha_schedule  $P_SCHEDULE      \
    --alpha           $P_ALPHA         \
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


ARCH=rs101
ARCHITECTURE=resnest101
AUGMENT=colorjitter
BATCH=32
TAG=puzzle/$DATASET-$ARCH-p
W_GROUP=$DATASET-pnoc-ow$OW_INIT-$OW-$OW_SCHEDULE-c$OC_TRAIN_MASK_T
W_TAGS="$DATASET,$ARCH,b:$BATCH,ac:$ACCUMULATE_STEPS,puzzle"
# run_training

ARCH=rs269
ARCHITECTURE=resnest269
BATCH=16
ACCUMULATE_STEPS=2
TAG=$DATASET-$ARCH-poc-b$BATCH-as$ACCUMULATE_STEPS@$OC_NAME
W_GROUP=$DATASET-pnoc-ow$OW_INIT-$OW-$OW_SCHEDULE-c$OC_TRAIN_MASK_T
W_TAGS="$DATASET,$ARCH,b:$BATCH,ac:$ACCUMULATE_STEPS,puzzle"
run_training
