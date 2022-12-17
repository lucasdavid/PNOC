

export PYTHONPATH=$(pwd)
export OMP_NUM_THREADS=8

PY=python
SOURCE=scripts/rw/train_affinity.py
DEVICE=cuda
DEVICES=0
WORKERS=8

# Dataset
IMAGE_SIZE=384      # 512
MIN_IMAGE_SIZE=340  # 320
MAX_IMAGE_SIZE=420  # 640

# Architecture
ARCHITECTURE=resnet50
BATCH_SIZE=8
LR=0.1


run() {
    echo "============================================================"
    echo "Experiment $TAG"
    echo "============================================================"

    CUDA_VISIBLE_DEVICES=$DEVICES    \
    WANDB_PROJECT=research-wsss-dev  \
    WANDB_TAGS="$DATASET,$ARCH,rw"   \
    $PY $SOURCE                      \
        --architecture $ARCHITECTURE \
        --tag          $TAG          \
        --batch_size   $BATCH_SIZE   \
        --image_size   $IMAGE_SIZE   \
        --min_image_size $MIN_IMAGE_SIZE \
        --max_image_size $MAX_IMAGE_SIZE \
        --dataset      $DATASET      \
        --lr           $LR           \
        --label_dir    $LABEL_DIR    \
        --data_dir     $DATA_DIR     \
        --num_workers  $WORKERS
}


DATASET=voc12
DATA_DIR=/home/ldavid/workspace/datasets/voc/VOCdevkit/VOC2012/
DOMAIN=train_aug
TAG=affnet@rs269-poc@pn-fgh@crf-10-gt-0.9@aff_fg=0.40_bg=0.10-dev
LABEL_DIR=./experiments/predictions/rw/affnet@rs269-poc@crf-10-gt-0.9@aff_fg=0.40_bg=0.10
run

# DATASET=coco14
# DATA_DIR=/home/ldavid/workspace/datasets/coco14/
# DOMAIN=train2014
# TAG=
# LABEL_DIR=
# run
