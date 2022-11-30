#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH -p sequana_gpu_shared
#SBATCH -J tr-dlv3
#SBATCH -o /scratch/lerdl/lucas.david/logs/dlv3/test-%j.out
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

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$(pwd)

PY=python3.9
SOURCE=train_segmentation.py
DATA_DIR=$SCRATCH/datasets/VOCdevkit/VOC2012/

# Architecture
ARCHITECTURE=resnest269
GROUP_NORM=true
DILATED=false
BATCH_SIZE=32
MODE=normal
LR=0.1

TAG=dlv3p-$MODE-gn@pn-fgh@rs269-poc
CRF_T=10

# MODE=fix
# TAG=DeepLabv3+@ResNeSt-269@Fix@GN

echo "==========\nInference"

CUDA_VISIBLE_DEVICES=0           \
$PY inference_segmentation.py    \
    --backbone $ARCHITECTURE     \
    --mode     $MODE             \
    --dilated  $DILATED          \
    --use_gn   $GROUP_NORM       \
    --tag      $TAG              \
    --domain   train             \
    --scale    0.5,1.0,1.5,2.0   \
    --iteration $CRF_T           \
    --data_dir $DATA_DIR         &

CUDA_VISIBLE_DEVICES=1           \
$PY inference_segmentation.py    \
    --backbone $ARCHITECTURE     \
    --mode     $MODE             \
    --dilated  $DILATED          \
    --use_gn   $GROUP_NORM       \
    --tag      $TAG              \
    --domain   val               \
    --scale    0.5,1.0,1.5,2.0   \
    --iteration $CRF_T           \
    --data_dir $DATA_DIR         &

wait
