#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH -p sequana_gpu_shared
#SBATCH -J irn-50
#SBATCH -o /scratch/lerdl/lucas.david/logs/irn/%j.out
#SBATCH --time=48:00:00

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

echo "[sdumont/sequana/classification/train-puzzle] started running at $(date +'%Y-%m-%d %H:%M:%S')."

nodeset -e $SLURM_JOB_NODELIST

cd $SCRATCH/PuzzleCAM/irn

module load sequana/current
module load gcc/7.4_sequana python/3.9.1_sequana cudnn/8.2_cuda-11.1_sequana
# module load gcc/7.4 python/3.9.1 cudnn/8.2_cuda-11.1

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$(pwd)

PY=python3.9
SOURCE=run_sample.py
LOGS_DIR=$SCRATCH/logs/irn
DATA_DIR=$SCRATCH/datasets/VOCdevkit/VOC2012/

TAG=irn@rs269-poc@pn-fgh@crf-10-gt-0.9@aff_fg=0.50_bg=0.05
ARCHITECTURE=resnest269
PROPOSALS=$SCRATCH/PuzzleCAM/experiments/predictions/affnet@rs269-poc@pn-fgh@crf-10-gt-0.9@aff_fg=0.50_bg=0.05
CAMS_DIR=$SCRATCH/PuzzleCAM/experiments/predictions/ResNeSt269@PuzzleOc@train@scale=0.5,1.0,1.5,2.0
CAM_FORMAT=puzzle

# Mining Inter-pixel Relations
# conf_fg_thres=0.30, type=float
# conf_bg_thres=0.05, type=float

# Inter-pixel Relation Network (IRNet
# irn_network="net.resnet50_irn"
# irn_crop_size=512, type=int
# irn_batch_size=32, type=int
# irn_num_epoches=3, type=int
# irn_learning_rate=0.1, type=float
# irn_weight_decay=1e-4, type=float

# Random Walk Params
# BETA=10
# EXP_TIMES=8, Hyper-parameter that controls the number of random walk iterations, the random walk is performed 2^{exp_times}.

CUDA_VISIBLE_DEVICES=0,1,2,3             \
    $PY $SOURCE                          \
    --log_name             $TAG          \
    --model_name           $ARCHITECTURE \
    --irn_weights_name     sess/$TAG.pth \
    --sem_seg_out_dir      "result/$TAG-seg" \
    --ins_seg_out_dir      "result/$TAG-ins" \
    --ir_label_out_dir     $PROPOSALS    \
    --cam_out_dir          $CAMS_DIR     \
    --cam_saved_format     $CAM_FORMAT   \
    --voc12_root           $DATA_DIR     \
    --train_cam_pass       false \
    --make_cam_pass        false \
    --eval_cam_pass        false \
    --cam_to_ir_label_pass false \
    --train_irn_pass       true  \
    --make_ins_seg_pass    false \
    --eval_ins_seg_pass    false \
    --make_sem_seg_pass    true  \
    --eval_sem_seg_pass    false
