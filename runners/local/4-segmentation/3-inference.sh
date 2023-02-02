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


run_inference() {
  echo "================================================="
  echo "Inference $TAG"
  echo "================================================="

  CUDA_VISIBLE_DEVICES=$DEVICES    \
  $PY $SOURCE                      \
      --backbone $ARCHITECTURE     \
      --mode     $MODE             \
      --dilated  $DILATED          \
      --use_gn   $GROUP_NORM       \
      --tag      $TAG              \
      --domain   $DOMAIN           \
      --scale    $SCALES           \
      --crf_t    $CRF_T            \
      --crf_gt_prob $CRF_GT_PROB   \
      --data_dir $DATA_DIR
}

# Architecture
ARCHITECTURE=resnest269
GROUP_NORM=true
DILATED=false
BATCH_SIZE=32
MODE=normal

SCALES=0.5,1.0,1.5,2.0
CRF_T=0
CRF_GT_PROB=0.7

# TAG=segmentation/dlv3p-normal-gn@pn-fgh@rs269-poc
# DOMAIN=train
# run_inference
# DOMAIN=val
# run_inference

# TAG=segmentation/d3p-normal-gn-sup
# DOMAIN=train
# run_inference
# DOMAIN=val
# run_inference

# TAG=segmentation/d3p@pn-ccamh@rs269apoc-ls0.1
# DOMAIN=train
# run_inference
# DOMAIN=val
# run_inference

TAG=segmentation/d3p-ls0.1@pn-ccamh@rs269apoc-ls0.1
CRF_T=1
CRF_GT_PROB=0.9
DOMAIN=test
run_inference
