

export PYTHONPATH=$(pwd)
export OMP_NUM_THREADS=8

PY=python
SOURCE=scripts/rw/inference.py
DEVICE=cuda
DEVICES=0
WORKERS=8

# Dataset
IMAGE_SIZE=512

run() {
    CUDA_VISIBLE_DEVICES=$DEVICES    \
    $PY $SOURCE                      \
        --architecture $ARCHITECTURE \
        --image_size   $IMAGE_SIZE   \
        --model_name   $MODEL_NAME   \
        --cam_dir      $CAM_DIR      \
        --domain       $DOMAIN       \
        --beta         10            \
        --exp_times    8             \
        --data_dir     $DATA_DIR
}


DATASET=voc12
DATA_DIR=/home/ldavid/workspace/datasets/voc/VOCdevkit/VOC2012/
DOMAIN=train_aug

ARCHITECTURE=resnest269
MODEL_NAME=affnet@rs269-poc@pn-fgh@crf-10-gt-0.9@aff_fg=0.40_bg=0.10
CAM_DIR=poc/ResNeSt269@PuzzleOc@train@scale=0.5,1.0,1.5,2.0
RW_DIR=rw/$MODEL_NAME@train@beta=10@exp_times=8@rw
run

# DATASET=coco14
# DATA_DIR=/home/ldavid/workspace/datasets/coco14/
# DOMAIN=train2014
# MODEL_NAME=
# CAM_DIR=
# RW_DIR=
# run
