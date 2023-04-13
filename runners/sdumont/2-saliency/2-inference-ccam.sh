#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH -p sequana_gpu_shared
#SBATCH -J mk-ccam
#SBATCH -o /scratch/lerdl/lucas.david/logs/ccam/mk-%j.out
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
# Train ResNeSt269 to perform multilabel classification
# task over Pascal VOC 2012 using OC-CSE strategy.
#

echo "[sdumont/sequana/saliency/inference] started running at $(date +'%Y-%m-%d %H:%M:%S')."

nodeset -e $SLURM_JOB_NODELIST

cd $SCRATCH/PuzzleCAM

module load sequana/current
module load gcc/7.4_sequana python/3.9.1_sequana cudnn/8.2_cuda-11.1_sequana
# module load gcc/7.4 python/3.9.1 cudnn/8.2_cuda-11.1


export PYTHONPATH=$(pwd)

PY=python3.9
SOURCE=scripts/ccam/inference.py
DEVICES=0,1,2,3

LOGS_DIR=$SCRATCH/logs/ccam

DATASET=voc12
DOMAIN=train
DATA_DIR=$SCRATCH/datasets/VOCdevkit/VOC2012/

ARCHITECTURE=resnest269
TRAINABLE_STEM=true
DILATED=false
MODE=normal
S4_OUT_FEATURES=1024
WEIGHTS=imagenet

run_inference() {
  CUDA_VISIBLE_DEVICES=$DEVICES $PY $SOURCE \
    --tag             $TAG                 \
    --dataset         $DATASET             \
    --domain          $DOMAIN              \
    --architecture    $ARCHITECTURE        \
    --dilated         $DILATED             \
    --stage4_out_features $S4_OUT_FEATURES \
    --mode            $MODE                \
    --weights         $WEIGHTS             \
    --trainable-stem  $TRAINABLE_STEM      \
    --pretrained      $PRETRAINED          \
    --data_dir        $DATA_DIR
}

run_crf() {
  $PY scripts/ccam/inference_crf.py                    \
    --experiment_name $TAG@train@scale=0.5,1.0,1.5,2.0 \
    --dataset         $DATASET                         \
    --domain          $DOMAIN                          \
    --threshold       $T                               \
    --crf_t           $CRF_T                           \
    --crf_gt_prob     $CRF_GT_PROB                     \
    --data_dir        $DATA_DIR
}

DOMAIN=train_aug

CRF_T=10
CRF_GT_PROB=0.9

# TAG=ccam-h-rs269@rs269-poc-fg0.4-b32-lr0.001
# PRETRAINED=./experiments/models/saliency/ccam-fg-hints@resnest269@rs269-poc@0.4@h1.0-e10-b32-lr0.001.pth
# run_inference

# TAG=saliency/voc12-ccamh-rs269@rs269poc@fg0.4-h1.0-e10-b32-lr0.001
# PRETRAINED=./experiments/models/$TAG.pth
# T=0.2
# run_inference
# run_crf

# TAG=saliency/voc12-ccamh-rs269@rs269poc-ls0.1-r3@fg0.4-h1.0-e10-b32-lr0.001
# PRETRAINED=./experiments/models/$TAG.pth
# T=0.2
# run_inference
# run_crf

# TAG=saliency/voc12-ccamh-rs269@rs269pnoc-ls0.1-r3@fg0.3-h1.0-e10-b32-lr0.001
# PRETRAINED=./experiments/models/$TAG.pth
# T=0.2
# run_inference
# run_crf

## =============================================================================
## # MS COCO 2014
##

DATASET=coco14
DATA_DIR=$SCRATCH/datasets/coco14/
DOMAIN=train2014
TAG=saliency/coco14-ccamh-rs269@rs269pnoc-lr0.05@rs269@b64-fg0.3-lr0.0005-b64
PRETRAINED=./experiments/models/$TAG.pth
T=0.2
run_inference
run_crf
