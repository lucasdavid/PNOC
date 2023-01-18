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
SOURCE=scripts/segmentation/inference.py

WORKERS=8
DEVICES=0  # 0,1,2,3

DATASET=voc12
DATA_DIR=/home/ldavid/workspace/datasets/voc/VOCdevkit/VOC2012
DOMAIN=train
# DATASET=coco14
# DATA_DIR=/home/ldavid/workspace/datasets/coco14
# DOMAIN=train2014

# Architecture
ARCHITECTURE=resnest269
GROUP_NORM=true
DILATED=false
MODE=normal

CRF_T=10

run_experiment() {
  CUDA_VISIBLE_DEVICES=$DEVICES             \
  $PY $SOURCE                               \
      --num_workers       $WORKERS          \
      --backbone          $ARCHITECTURE     \
      --mode              $MODE             \
      --dilated           $DILATED          \
      --use_gn            $GROUP_NORM       \
      --tag               $TAG              \
      --dataset           $DATASET          \
      --domain            $DOMAIN           \
      --iteration         $CRF_T            \
      --data_dir          $DATA_DIR
}

TAG=segmentation/dlv3p-normal-gn@pn-fgh@rs269-poc
MASKS_DIR=$DATA_DIR/SegmentationClass
DOMAIN=test
run_experiment
