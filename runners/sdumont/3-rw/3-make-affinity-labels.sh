#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH -p nvidia_long
#SBATCH -J mk-aff
#SBATCH -o /scratch/lerdl/lucas.david/logs/puzzle/affinitynet/mk-%j.out
#SBATCH --time=36:00:00

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

echo "[voc12/puzzle/train.sequana] started running at $(date +'%Y-%m-%d %H:%M:%S')."

nodeset -e $SLURM_JOB_NODELIST

cd $SCRATCH/PuzzleCAM

# module load sequana/current
# module load gcc/7.4_sequana python/3.9.1_sequana cudnn/8.2_cuda-11.1_sequana
module load gcc/7.4 python/3.9.1 cudnn/8.2_cuda-11.1

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$(pwd)

PY=python3.9
SOURCE=make_affinity_labels.py
DATA_DIR=$SCRATCH/datasets/VOCdevkit/VOC2012/

ARCHITECTURE=resnest269
DILATED=false

DOMAIN=train_aug

CAMS_DIR=./experiments/predictions/ResNeSt269@PuzzleOc@train@scale=0.5,1.0,1.5,2.0/
SAL_DIR=./experiments/predictions/saliency/poolnet@ccam-fgh@rs269@rs269-poc/

FG=0.4
BG=0.1
CRF_T=10
CRF_GT=0.9

TAG=affnet@rs269-poc@pn-fgh@crf-$CRF_T-gt-$CRF_GT

$PY $SOURCE               \
    --tag     $TAG        \
    --domain  $DOMAIN     \
    --fg_threshold $FG    \
    --bg_threshold $BG    \
    --crf_t $CRF_T        \
    --crf_gt_prob $CRF_GT \
    --cams_dir $CAMS_DIR  \
    --sal_dir  $SAL_DIR   \
    --data_dir $DATA_DIR  &

# FG=0.5
# BG=0.05
# TAG=affnet@rs269-poc@pn-fgh@crf-$CRF_T-gt-$CRF_GT

# $PY $SOURCE               \
#     --tag     $TAG        \
#     --domain  $DOMAIN     \
#     --fg_threshold $FG    \
#     --bg_threshold $BG    \
#     --crf_t $CRF_T        \
#     --crf_gt_prob $CRF_GT \
#     --cams_dir $CAMS_DIR  \
#     --sal_dir  $SAL_DIR   \
#     --data_dir $DATA_DIR  &

wait
