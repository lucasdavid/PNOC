#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH -p sequana_gpu_shared
#SBATCH -J inf-dlv3
#SBATCH -o /scratch/lerdl/lucas.david/logs/dlv3/inf-%j.out
#SBATCH --time=4:00:00

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

echo "[voc12/segmentation/evaluate-deeplabv3] started running at $(date +'%Y-%m-%d %H:%M:%S')."

nodeset -e $SLURM_JOB_NODELIST

cd $SCRATCH/PuzzleCAM

module load sequana/current
module load gcc/7.4_sequana python/3.9.1_sequana cudnn/8.2_cuda-11.1_sequana


export PYTHONPATH=$(pwd)

PY=python3.9
SOURCE=scripts/segmentation/inference.py
DEVICES=0,1,2,3

DATASET=voc12
DATA_DIR=$SCRATCH/datasets/VOCdevkit/VOC2012/
DOMAIN=train
# DATASET=coco14
# DATA_DIR=$SCRATCH/datasets/coco14/
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
