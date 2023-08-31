#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH -p sequana_gpu_shared
#SBATCH -J dv2-voc12_513px
#SBATCH -o /scratch/lerdl/lucas.david/logs/%j-dv2-voc12_513px.out
#SBATCH --time=48:00:00

# Copyright 2023 Lucas Oliveira David
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
# Train a DeepLabV2 model to perform semantic segmentation.
#

if [[ "`hostname`" == "sdumont8"* ]]; then
  ENV=sdumont
  WORK_DIR=$SCRATCH/deeplab-pytorch

  ### Sdumont
  PY=python3.8
  PIP=pip3.8
  DEVICES=0,1,2,3

  # export OMP_NUM_THREADS=4
  nodeset -e $SLURM_JOB_NODELIST
  module load sequana/current
  module load gcc/7.4_sequana python/3.8.2_sequana cudnn/8.2_cuda-11.1_sequana
elif [[ "`hostname`" == "sdumont3"* ]]; then
  ENV=sdumont
  WORK_DIR=$SCRATCH/deeplab-pytorch

  ### Sdumont
  PY=python
  PIP=pip
  DEVICES=0,1

  # export OMP_NUM_THREADS=4
  nodeset -e $SLURM_JOB_NODELIST
  module load gcc/7.4 cudnn/8.2_cuda-11.3

  source $SCRATCH/envs/torch1.12/bin/activate
else
  ENV=local
  WORK_DIR=$HOME/workspace/repos/research/wsss/deeplab-pytorch

  ### Local
  PY=python
  PIP=pip
  DEVICES=0
fi

cd $SCRATCH/deeplab-pytorch

# echo "Installing dependencies... "
# $PIP -qq install click tqdm addict joblib omegaconf opencv-python torchnet tensorboard
# echo "Done."

EXP_ID=voc12_512px  # voc12_418px  # voc12  # (321px)

IMAGES_DIR=data/datasets/voc12/VOCdevkit/VOC2012/JPEGImages
TXT_FILE=data/datasets/voc12/VOCdevkit/VOC2012/ImageSets/Segmentation/test.txt
MODEL_PATH=data/models/$EXP_ID/deeplabv2_resnet101_msc/train_aug/checkpoint_final.pth
PRED_DIR=data/features/$EXP_ID/deeplabv2_resnet101_msc/test/preds

CUDA_VISIBLE_DEVICES=$DEVICES $PY main.py train  --config-path configs/$EXP_ID.yaml
CUDA_VISIBLE_DEVICES=$DEVICES $PY main.py test   --config-path configs/$EXP_ID.yaml --model-path $MODEL_PATH
CUDA_VISIBLE_DEVICES=$DEVICES $PY main.py crf    --config-path configs/$EXP_ID.yaml

CUDA_VISIBLE_DEVICES=$DEVICES $PY demo_testvoc.py single --config-path configs/$EXP_ID.yaml --model-path $MODEL_PATH \
  --img-dir $IMAGES_DIR --txt-file $TXT_FILE --save-dir $PRED_DIR --crf
