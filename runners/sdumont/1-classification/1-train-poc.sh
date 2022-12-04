#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH -p sequana_gpu_shared
#SBATCH -J tr-poc
#SBATCH -o /scratch/lerdl/lucas.david/logs/puzzle/poc-%j.out
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
SOURCE=scripts/cam/train_poc.py
WORKERS=16

# DATASET=voc12
# DATA_DIR=$SCRATCH/datasets/VOCdevkit/VOC2012/
DATASET=coco14
DATA_DIR=$SCRATCH/datasets/coco14/


ARCH=rs101
ARCHITECTURE=resnest101
TRAINABLE_STEM=true
DILATED=false
MODE=normal
REGULAR=none

IMAGE_SIZE=512
EPOCHS=15
BATCH=32

OC_PRETRAINED=experiments/models/coco14-rs101.pth
OC_ARCHITECTURE=$ARCHITECTURE
OC_REGULAR=none
OC_NAME=rs101

OC_STRATEGY=random
OC_F_MOMENTUM=0.8
OC_F_GAMMA=5.0
# Schedule
P_INIT=0.0
P_ALPHA=4.0
P_SCHEDULE=0.5
OC_INIT=0.3
OC_ALPHA=1.0
OC_SCHEDULE=1.0

AUGMENT=colorjitter
CUTMIX=0.5
MIXUP=1.
LABELSMOOTHING=0  # 0.1

TAG=$DATASET-$ARCH-poc@$OC_NAME
CUDA_VISIBLE_DEVICES=0,1,2,3             \
$PY $SOURCE                              \
    --tag               $TAG             \
    --num_workers       $WORKERS         \
    --batch_size        $BATCH           \
    --architecture      $ARCHITECTURE    \
    --dilated           $DILATED         \
    --mode              $MODE            \
    --trainable-stem    $TRAINABLE_STEM  \
    --regularization    $REGULAR         \
    --oc-architecture   $OC_ARCHITECTURE \
    --oc-pretrained     $OC_PRETRAINED   \
    --oc-regularization $OC_REGULAR      \
    --image_size        $IMAGE_SIZE      \
    --min_image_size    320              \
    --max_image_size    640              \
    --augment           $AUGMENT         \
    --cutmix_prob       $CUTMIX          \
    --mixup_prob        $MIXUP           \
    --label_smoothing   $LABELSMOOTHING  \
    --max_epoch         $EPOCHS          \
    --alpha             $P_ALPHA         \
    --alpha_init        $P_INIT          \
    --alpha_schedule    $P_SCHEDULE      \
    --oc-alpha          $OC_ALPHA        \
    --oc-alpha-init     $OC_INIT         \
    --oc-alpha-schedule $OC_SCHEDULE     \
    --oc-strategy       $OC_STRATEGY     \
    --oc-focal-momentum $OC_F_MOMENTUM   \
    --oc-focal-gamma    $OC_F_GAMMA      \
    --dataset           $DATASET         \
    --data_dir          $DATA_DIR


# DOMAIN=train

# CUDA_VISIBLE_DEVICES=0                   \
#     $PY inference_classification.py      \
#     --architecture      $ARCHITECTURE    \
#     --regularization    $REG             \
#     --dilated           $DILATED         \
#     --trainable-stem    $TRAINABLE_STEM  \
#     --mode              $MODE            \
#     --tag               $TAG             \
#     --domain            $DOMAIN          \
#     --data_dir          $DATA_DIR

# $PY evaluate.py                                        \
#   --experiment_name "$TAG@train@scale=0.5,1.0,1.5,2.0" \
#   --domain $DOMAIN                                     \
#   --gt_dir "$DATA_DIR"SegmentationClass                \
#   --min_th 0.05 \
#   --max_th 0.9 \
#   --step_th 0.05
