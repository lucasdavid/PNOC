

export PYTHONPATH=$(pwd)
# export OMP_NUM_THREADS=4

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
DOMAIN=train

ARCHITECTURE=resnest269
# MODEL_NAME=affnet@rs269-poc@pn-fgh@crf-10-gt-0.9@aff_fg=0.40_bg=0.10
# CAM_DIR=poc/ResNeSt269@PuzzleOc@train@scale=0.5,1.0,1.5,2.0
# RW_DIR=rw/$MODEL_NAME@train@beta=10@exp_times=8@rw
# run

DOMAIN=val

MODEL_NAME=rw/AffinityNet@ResNeSt-269@Puzzle
CAM_DIR=puzzle/ResNeSt269@Puzzle@optimal@$DOMAIN@scale=0.5,1.0,1.5,2.0
RW_DIR=$MODEL_NAME@$DOMAIN@beta=10@exp_times=8@rw
run

MODEL_NAME=rw/AffinityNet@resnest269@puzzlerep@aff_fg=0.40_bg=0.10
CAM_DIR=puzzle/resnest269@puzzlerep@$DOMAIN@scale=0.5,1.0,1.5,2.0
RW_DIR=$MODEL_NAME@$DOMAIN@beta=10@exp_times=8@rw
run

MODEL_NAME=rw/AffinityNet@ResNeSt269@PuzzleOc
CAM_DIR=poc/ResNeSt269@PuzzleOc@$DOMAIN@scale=0.5,1.0,1.5,2.0
RW_DIR=$MODEL_NAME@$DOMAIN@beta=10@exp_times=8@rw
run

MODEL_NAME=rw/voc12-an@rs269poc-ls0.1@fg0.4-bg0.1-crf10-gt0.9@aff_fg=0.40_bg=0.10
CAM_DIR=poc/voc12-rs269-poc-ls0.1@rs269ra-r3@$DOMAIN@scale=0.5,1.0,1.5,2.0
RW_DIR=$MODEL_NAME@$DOMAIN@beta=10@exp_times=8@rw
run

MODEL_NAME=rw/voc12-an@ccamh@rs269poc-ls0.1@fg0.4-bg0.1-crf10-gt0.9@aff_fg=0.40_bg=0.10
CAM_DIR=poc/voc12-rs269-poc-ls0.1@rs269ra-r3@$DOMAIN@scale=0.5,1.0,1.5,2.0
RW_DIR=$MODEL_NAME@$DOMAIN@beta=10@exp_times=8@rw
run

MODEL_NAME=rw/voc12-an@rs269apoc-ls0.1@fg0.3-bg0.1-crf10-gt0.7@aff_fg=0.30_bg=0.10
CAM_DIR=apoc/voc12-rs269-apoc-ls0.1-ow0.0-1.0-1.0-cams-0.2-octis1-amp@rs269ra-r3@$DOMAIN@scale=0.5,1.0,1.5,2.0
RW_DIR=$MODEL_NAME@$DOMAIN@beta=10@exp_times=8@rw
run

# MODEL_NAME=rw/voc12-an@rs269apoc-ls0.1@fg0.3-bg0.1-crf10-gt0.7@aff_fg=0.30_bg=0.10

## ===================================
## COCO 14

# DATASET=coco14
# DATA_DIR=/home/ldavid/workspace/datasets/coco14/
# DOMAIN=train2014
# MODEL_NAME=
# CAM_DIR=
# RW_DIR=
# run
