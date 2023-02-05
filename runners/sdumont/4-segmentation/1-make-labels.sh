#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH -p nvidia_long
#SBATCH -J mk-pseudo-segm
#SBATCH -o /scratch/lerdl/lucas.david/logs/mk-pseudo-segm-%j.out
#SBATCH --time=03:00:00

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

echo "[voc12/segmentation/make-labels] started running at $(date +'%Y-%m-%d %H:%M:%S')."

nodeset -e $SLURM_JOB_NODELIST

cd $SCRATCH/PuzzleCAM

# module load sequana/current
# module load gcc/7.4_sequana python/3.9.1_sequana cudnn/8.2_cuda-11.1_sequana
module load gcc/7.4 python/3.9.1 cudnn/8.2_cuda-11.1


export PYTHONPATH=$(pwd)

PY=python3.9
SOURCE=scripts/segmentation/make_pseudo_labels.py

DATASET=voc12
DATA_DIR=$SCRATCH/datasets/VOCdevkit/VOC2012/
DOMAIN=train_aug
# DATASET=coco14
# DATA_DIR=$SCRATCH/datasets/coco14/
# DOMAIN=train2014

CAMS_DIR=./experiments/predictions/ResNeSt269@PuzzleOc@train@scale=0.5,1.0,1.5,2.0/
# SAL_DIR=./experiments/predictions/saliency/poolnet@ccam-fgh@rs269@rs269-poc/

THRESHOLD=0.25
CRF_T=1
CRF_GT=0.9

run_inference() {
  $PY $SOURCE                      \
      --experiment_name $TAG       \
      --domain          $DOMAIN    \
      --threshold       $THRESHOLD \
      --crf_t           $CRF_T     \
      --crf_gt_prob     $CRF_GT    \
      --data_dir        $DATA_DIR
}

TAG=rw/voc12-an@ccamh@rs269pnoc-ls0.1@fg0.3-bg0.1-crf10-gt0.7@train@beta=10@exp_times=8@rw
DOMAIN=train_aug
run_inference

# TAG=rw/voc12-an@ccamh@rs269pnoc-ls0.1@fg0.3-bg0.1-crf10-gt0.7@val@beta=10@exp_times=8@rw
# DOMAIN=val
# run_inference
