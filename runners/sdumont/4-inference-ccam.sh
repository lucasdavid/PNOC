#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH -p sequana_gpu_shared
#SBATCH -J mk-ccam
#SBATCH -o /scratch/lerdl/lucas.david/logs/ccam/mk-%j.out
#SBATCH --time=04:00:00

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

echo "[voc12/puzzle/train.sequana] started running at $(date +'%Y-%m-%d %H:%M:%S')."

nodeset -e $SLURM_JOB_NODELIST

cd $SCRATCH/PuzzleCAM

module load sequana/current
module load gcc/7.4_sequana python/3.9.1_sequana cudnn/8.2_cuda-11.1_sequana
# module load gcc/7.4 python/3.9.1 cudnn/8.2_cuda-11.1

PY=python3.9
SOURCE=ccam_inference.py

LOGS_DIR=$SCRATCH/logs/ccam
DATA_DIR=$SCRATCH/datasets/VOCdevkit/VOC2012/

WORKERS=8

DILATED=false
TRAINABLE_STEM=true
MODE=normal

DOMAIN=train_aug


TAG=ccam-fgh@rs269@rs269-poc
ARCHITECTURE=resnest269
S4_OUT_FEATURES=1024
WEIGHTS=./experiments/models/ccam-fg-hints@resnest269@rs269-poc@0.4@h1.0-e10-b32-lr0.001.pth
# THRESHOLD=0.3
# CRF=10

CUDA_VISIBLE_DEVICES=0 $PY $SOURCE       \
  --tag             $TAG                 \
  --domain          $DOMAIN              \
  --num_workers     $WORKERS             \
  --architecture    $ARCHITECTURE        \
  --dilated         $DILATED             \
  --stage4_out_features $S4_OUT_FEATURES \
  --mode            $MODE                \
  --trainable-stem  $TRAINABLE_STEM      \
  --pretrained      $WEIGHTS             \
  --data_dir        $DATA_DIR

EXP=$TAG@train@scale=0.5,1.0,1.5,2.0

python3.9 ccam_inference_crf.py \
  --experiment_name $EXP        \
  --domain          train_aug   \
  --threshold       0.5         \
  --crf_iteration   10          \
  --data_dir        $DATA_DIR
