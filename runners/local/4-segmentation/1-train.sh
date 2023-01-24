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
# Segmentation/Train DeepLabV3+
#

export PYTHONPATH=$(pwd)

PY=python
SOURCE=scripts/segmentation/train.py

WORKERS=8
DEVICES=0  # 0,1,2,3

DATASET=voc12
DATA_DIR=/home/ldavid/workspace/datasets/voc/VOCdevkit/VOC2012
DOMAIN=train
# DATASET=coco14
# DATA_DIR=/home/ldavid/workspace/datasets/coco14
# DOMAIN=train2014

# Architecture
ARCHITECTURE=resnet50
GROUP_NORM=true
DILATED=false
MODE=normal

LR=0.007

EPOCHS=50
BATCH_SIZE=8
ACCUMULATE_STEPS=1
MIXED_PRECISION=true

IMAGE_SIZE=64      # 512
MIN_IMAGE_SIZE=32  # 256
MAX_IMAGE_SIZE=128 # 1024

AUGMENT=colorjitter
CUTMIX=0.5
MIXUP=1.
LABELSMOOTHING=0  # 0.1

run_experiment() {
  WANDB_TAGS="$DATASET,$ARCH,segmentation,ls:$LABEL_SMOOTHING"  \
  CUDA_VISIBLE_DEVICES=$DEVICES             \
  $PY $SOURCE                               \
      --tag               $TAG              \
      --num_workers       $WORKERS          \
      --lr                $LR               \
      --epochs            $EPOCHS           \
      --batch_size        $BATCH_SIZE       \
      --accumulate_steps  $ACCUMULATE_STEPS \
      --mixed_precision   $MIXED_PRECISION  \
      --architecture      $ARCHITECTURE     \
      --dilated           $DILATED          \
      --mode              $MODE             \
      --use_gn            $GROUP_NORM       \
      --image_size        $IMAGE_SIZE       \
      --min_image_size    $MIN_IMAGE_SIZE   \
      --max_image_size    $MAX_IMAGE_SIZE   \
      --augment           $AUGMENT          \
      --cutmix_prob       $CUTMIX           \
      --mixup_prob        $MIXUP            \
      --label_smoothing   $LABELSMOOTHING   \
      --data_dir          $DATA_DIR         \
      --masks_dir         $MASKS_DIR
}

TAG=dlv3p-$MODE-gn-supervised
MASKS_DIR=$DATA_DIR/SegmentationClass
run_experiment

# TAG=dlv3p-$MODE-gn@pn-fgh@rs269apoc
# MASKS_DIR=./experiments/predictions/rw/affnet@rs269-poc@pn-fgh@crf-10-gt-0.9@aff_fg=0.40_bg=0.10@train@beta=10@exp_times=8@rw@crf=1
