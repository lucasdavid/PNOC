#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH -p sequana_gpu_shared
#SBATCH -J tr-apoc
#SBATCH -o /scratch/lerdl/lucas.david/logs/puzzle/apoc-%j.out
#SBATCH --time=72:00:00

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
SOURCE=scripts/cam/train_apoc.py
WORKERS=8
DEVICES=0,1,2,3

DATASET=voc12
DATA_DIR=$SCRATCH/datasets/VOCdevkit/VOC2012/
DOMAIN=train
# DATASET=coco14
# DATA_DIR=$SCRATCH/datasets/coco14/
# DOMAIN=train2014


ARCH=rs269
ARCHITECTURE=resnest269
TRAINABLE_STEM=true
DILATED=false
MODE=normal
REGULAR=none

IMAGE_SIZE=512
MIN_IMAGE_SIZE=320
MAX_IMAGE_SIZE=640
EPOCHS=15
BATCH=16
ACCUMULATE_STEPS=1
MIXED_PRECISION=true

OC_NAME=rs269poc
OC_PRETRAINED=experiments/models/ResNeSt269@PuzzleOc.pth
OC_ARCHITECTURE=$ARCHITECTURE
OC_REGULAR=none
OC_TRAIN_MASKS=features
OC_TRAIN_MASK_T=0.2
OC_TRAIN_INT_STEPS=1

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

run_training () {
    echo "============================================================"
    echo "Experiment $TAG"
    echo "============================================================"
    CUDA_VISIBLE_DEVICES=$DEVICES            \
    $PY $SOURCE                              \
        --tag               $TAG             \
        --num_workers       $WORKERS         \
        --batch_size        $BATCH           \
        --accumulate_steps  $ACCUMULATE_STEPS \
        --mixed_precision   $MIXED_PRECISION \
        --architecture      $ARCHITECTURE    \
        --dilated           $DILATED         \
        --mode              $MODE            \
        --trainable-stem    $TRAINABLE_STEM  \
        --regularization    $REGULAR         \
        --oc-architecture   $OC_ARCHITECTURE \
        --oc-pretrained     $OC_PRETRAINED   \
        --oc-regularization $OC_REGULAR      \
        --image_size        $IMAGE_SIZE      \
        --min_image_size    $MIN_IMAGE_SIZE  \
        --max_image_size    $MAX_IMAGE_SIZE  \
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
        --ow                $OW              \
        --ow-init           $OW_INIT         \
        --ow-schedule       $OW_SCHEDULE     \
        --oc-train-masks    $OC_TRAIN_MASKS  \
        --oc_train_mask_t   $OC_TRAIN_MASK_T \
        --oc-train-interval-steps $OC_TRAIN_INT_STEPS \
        --dataset           $DATASET         \
        --data_dir          $DATA_DIR
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
    --data_dir          $DATA_DIR        &
}


# OC_NAME=rs269ra
# OC_PRETRAINED=experiments/models/resnest269@randaug.pth
# OC_ARCHITECTURE=resnest269
# LABELSMOOTHING=0.1
# OW=1.0
# OW_INIT=0.3
# OW_SCHEDULE=0.5
# OC_TRAIN_MASKS=features
# TAG=$DATASET-$ARCH-apoc-ls$LABELSMOOTHING-ow$OW_INIT-$OW-$OW_SCHEDULE-$OC_TRAIN_MASKS-amp@$OC_NAME-r1
# run_training
# TAG=$DATASET-$ARCH-apoc-ls$LABELSMOOTHING-ow$OW_INIT-$OW-$OW_SCHEDULE-$OC_TRAIN_MASKS-amp@$OC_NAME-r2
# run_training
# TAG=$DATASET-$ARCH-apoc-ls$LABELSMOOTHING-ow$OW_INIT-$OW-$OW_SCHEDULE-$OC_TRAIN_MASKS-amp@$OC_NAME-r3
# run_training

# DEVICES=0
# TAG=$DATASET-$ARCH-apoc-ls$LABELSMOOTHING-ow$OW_INIT-$OW-$OW_SCHEDULE-$OC_TRAIN_MASKS-amp@$OC_NAME-r1
# run_inference
# DEVICES=2
# TAG=$DATASET-$ARCH-apoc-ls$LABELSMOOTHING-ow$OW_INIT-$OW-$OW_SCHEDULE-$OC_TRAIN_MASKS-amp@$OC_NAME-r2
# run_inference
# DEVICES=3
# TAG=$DATASET-$ARCH-apoc-ls$LABELSMOOTHING-ow$OW_INIT-$OW-$OW_SCHEDULE-$OC_TRAIN_MASKS-amp@$OC_NAME-r3
# run_inference
# wait

# OC_NAME=rs269ra
# OC_PRETRAINED=experiments/models/resnest269@randaug.pth
# OC_ARCHITECTURE=resnest269
# LABELSMOOTHING=0.1
# OW=0.5
# OW_INIT=0.0
# OW_SCHEDULE=1.0
# OC_TRAIN_MASKS=features
# OC_TRAIN_INT_STEPS=5
# TAG=$DATASET-$ARCH-apoc-ls$LABELSMOOTHING-ow$OW_INIT-$OW-$OW_SCHEDULE-$OC_TRAIN_MASKS-octrint$OC_TRAIN_INT_STEPS-amp@$OC_NAME-r1
# run_training
# TAG=$DATASET-$ARCH-apoc-ls$LABELSMOOTHING-ow$OW_INIT-$OW-$OW_SCHEDULE-$OC_TRAIN_MASKS-octrint$OC_TRAIN_INT_STEPS-amp@$OC_NAME-r2
# run_training
# TAG=$DATASET-$ARCH-apoc-ls$LABELSMOOTHING-ow$OW_INIT-$OW-$OW_SCHEDULE-$OC_TRAIN_MASKS-octrint$OC_TRAIN_INT_STEPS-amp@$OC_NAME-r3
# run_training


OC_NAME=rs269ra
OC_PRETRAINED=experiments/models/resnest269@randaug.pth
OC_ARCHITECTURE=resnest269
LABELSMOOTHING=0.1
OW=0.5
OW_INIT=0.0
OW_SCHEDULE=1.0
OC_TRAIN_MASKS=cams
OC_TRAIN_MASK_T=0.2
OC_TRAIN_INT_STEPS=1
TAG=$DATASET-$ARCH-apoc-ls$LABELSMOOTHING-ow$OW_INIT-$OW-$OW_SCHEDULE-$OC_TRAIN_MASKS-$OC_TRAIN_MASK_T-octis$OC_TRAIN_INT_STEPS-amp@$OC_NAME-r4
run_training
OW=1.0
OC_TRAIN_MASK_T=0.2
TAG=$DATASET-$ARCH-apoc-ls$LABELSMOOTHING-ow$OW_INIT-$OW-$OW_SCHEDULE-$OC_TRAIN_MASKS-$OC_TRAIN_MASK_T-octis$OC_TRAIN_INT_STEPS-amp@$OC_NAME-r4
run_training
OC_TRAIN_INT_STEPS=5
OC_TRAIN_MASK_T=0.2
TAG=$DATASET-$ARCH-apoc-ls$LABELSMOOTHING-ow$OW_INIT-$OW-$OW_SCHEDULE-$OC_TRAIN_MASKS-$OC_TRAIN_MASK_T-octis$OC_TRAIN_INT_STEPS-amp@$OC_NAME-r4
run_training

# DEVICES=0
# OC_TRAIN_MASK_T=0.2
# TAG=$DATASET-$ARCH-apoc-ls$LABELSMOOTHING-ow$OW_INIT-$OW-$OW_SCHEDULE-$OC_TRAIN_MASKS-$OC_TRAIN_MASK_T-amp@$OC_NAME-r1
# run_inference
# DEVICES=1
# OC_TRAIN_MASK_T=0.3
# TAG=$DATASET-$ARCH-apoc-ls$LABELSMOOTHING-ow$OW_INIT-$OW-$OW_SCHEDULE-$OC_TRAIN_MASKS-$OC_TRAIN_MASK_T-amp@$OC_NAME-r1
# run_inference
# DEVICES=2
# OC_TRAIN_MASK_T=0.4
# TAG=$DATASET-$ARCH-apoc-ls$LABELSMOOTHING-ow$OW_INIT-$OW-$OW_SCHEDULE-$OC_TRAIN_MASKS-$OC_TRAIN_MASK_T-amp@$OC_NAME-r1
# run_inference
# wait
