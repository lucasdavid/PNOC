#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH -p sequana_gpu_shared
#SBATCH -J tr-ccam
#SBATCH -o /scratch/lerdl/lucas.david/logs/ccam/mk-pn-%j.out
#SBATCH --time=01:00:00

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
# Train PoolNet over the VOC12 or COCO14 dataset.
#

echo "[sdumont/sequana/saliency/train-poolnet] started running at $(date +'%Y-%m-%d %H:%M:%S')."

nodeset -e $SLURM_JOB_NODELIST

cd $SCRATCH/PuzzleCAM/poolnet

module load sequana/current
module load gcc/7.4_sequana python/3.9.1_sequana cudnn/8.2_cuda-11.1_sequana
# module load gcc/7.4 python/3.9.1 cudnn/8.2_cuda-11.1
# export PYTHONPATH=$(pwd)

PY=python3.9
SOURCE=main_custom.py

DATASET=voc12
DOMAIN=train_aug
DATA_DIR=$SCRATCH/datasets/VOCdevkit/VOC2012/
# DATASET=coco14
# DOMAIN=train2014
# DATA_DIR=$SCRATCH/datasets/coco14/

LOGS_DIR=$SCRATCH/logs/ccam

ARCHITECTURE=resnet  # resnet vgg

# TAG=pn@ccamh-rs269@rs269poc
# PSEUDO_CUES=$SCRATCH/PuzzleCAM/experiments/predictions/saliency/voc12-ccamh-rs269@rs269poc@fg0.4-h1.0-e10-b32-lr0.001@train@scale=0.5,1.0,1.5,2.0@t=0.2@crf=10/
# TAG=pn@ccamh-rs269@rs269poc-ls0.1
# PSEUDO_CUES=$SCRATCH/PuzzleCAM/experiments/predictions/saliency/voc12-ccamh-rs269@rs269poc-ls0.1-r3@fg0.4-h1.0-e10-b32-lr0.001@train@scale=0.5,1.0,1.5,2.0@t=0.2@crf=10/
# TAG=pn@ccamh-rs269@rs269pnoc-ls0.1
# PSEUDO_CUES=$SCRATCH/PuzzleCAM/experiments/predictions/saliency/voc12-ccamh-rs269@rs269pnoc-ls0.1-r3@fg0.3-h1.0-e10-b32-lr0.001@train@scale=0.5,1.0,1.5,2.0@t=0.2@crf=10/

poolnet_training() {
  cd $SCRATCH/PuzzleCAM/poolnet

  CUDA_VISIBLE_DEVICES=0 $PY $SOURCE  \
    --arch        $ARCHITECTURE       \
    --mode        "train"             \
    --dataset     $DATASET            \
    --train_root  $DATA_DIR           \
    --train_list  $DOMAIN             \
    --pseudo_root $PSEUDO_CUES

}


poolnet_inference() {
  cd $SCRATCH/PuzzleCAM/poolnet

  CUDA_VISIBLE_DEVICES=0 $PY $SOURCE  \
    --arch        $ARCHITECTURE       \
    --mode        "test"              \
    --model       $MODEL_CKPT         \
    --dataset     $DATASET            \
    --train_list  $DOMAIN             \
    --train_root  $DATA_DIR           \
    --pseudo_root $PSEUDO_CUES        \
    --sal_folder ./results/$TAG

}


MODEL_CKPT=./results/run-1/models/epoch_9.pth
poolnet_training
poolnet_inference

# cp ./results/run-1/models/epoch_9.pth ../experiments/models/saliency/$TAG.pth
mv ./results/$TAG ../experiments/predictions/saliency/
