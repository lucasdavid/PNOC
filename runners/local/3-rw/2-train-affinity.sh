

export PYTHONPATH=$(pwd)
export OMP_NUM_THREADS=8

PY=python
SOURCE=scripts/rw/train_affinity.py
DEVICE=cuda
DEVICES=0
WORKERS=8

# Dataset
IMAGE_SIZE=512
MIN_IMAGE_SIZE=320
MAX_IMAGE_SIZE=640

# Architecture
ARCHITECTURE=resnest269
ARCH=rs269
BATCH_SIZE=32
LR=0.1


run_training() {
    echo "============================================================"
    echo "Experiment $TAG"
    echo "============================================================"

    CUDA_VISIBLE_DEVICES=$DEVICES    \
    WANDB_TAGS="$DATASET,$ARCH,rw"   \
    $PY $SOURCE                      \
        --architecture $ARCHITECTURE \
        --tag          $TAG       \
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

run_inference() {
  CUDA_VISIBLE_DEVICES=$DEVICES    \
  $PY scripts/rw/inference.py      \
      --architecture $ARCHITECTURE \
      --model_name $TAG            \
      --cam_dir $CAMS_DIR          \
      --beta 10                    \
      --exp_times 8                \
      --image_size $IMAGE_SIZE     \
      --dataset $DATASET           \
      --domain $DOMAIN             \
      --data_dir $DATA_DIR
}


DATASET=voc12
DATA_DIR=$SCRATCH/datasets/VOCdevkit/VOC2012/
DOMAIN=train
# DATASET=coco14
# DATA_DIR=$SCRATCH/datasets/coco14/
# DOMAIN=train2014


# TAG=rw/AffinityNet@ResNeSt-269@Puzzle
# CAMS_DIR=./experiments/predictions/puzzle/ResNeSt269@Puzzle@optimal@train@scale=0.5,1.0,1.5,2.0
# LABEL_DIR=./experiments/predictions/affinity/ResNeSt269@Puzzle@optimal@train@scale=0.5,1.0,1.5,2.0@aff_fg=0.40_bg=0.10
# run_training
# run_inference

# TAG=rw/voc12-an@rs269poc-ls0.1@fg0.4-bg0.1-crf10-gt0.9@aff_fg=0.40_bg=0.10
# CAMS_DIR=./experiments/predictions/poc/voc12-rs269-poc-ls0.1@rs269ra-r3@train@scale=0.5,1.0,1.5,2.0
# LABEL_DIR=./experiments/predictions/voc12-an@rs269poc-ls0.1@fg0.4-bg0.1-crf10-gt0.9@aff_fg=0.40_bg=0.10
# run_training
# run_inference

# TAG=rw/voc12-an@rs269pnoc-ls0.1@fg0.3-bg0.1-crf10-gt0.9@aff_fg=0.30_bg=0.10
# CAMS_DIR=./experiments/predictions/pnoc/voc12-rs269-pnoc-ls0.1-ow0.0-1.0-1.0-cams-0.2-octis1-amp@rs269ra-r3@train@scale=0.5,1.0,1.5,2.0
# LABEL_DIR=./experiments/predictions/voc12-an@rs269pnoc-ls0.1@fg0.3-bg0.1-crf10-gt0.9@aff_fg=0.30_bg=0.10
# run_training
# run_inference

# TAG=rw/voc12-an@ccamh@rs269poc-ls0.1@fg0.4-bg0.1-crf10-gt0.9@aff_fg=0.40_bg=0.10
# CAMS_DIR=./experiments/predictions/poc/voc12-rs269-poc-ls0.1@rs269ra-r3@train@scale=0.5,1.0,1.5,2.0
# LABEL_DIR=./experiments/predictions/voc12-an@ccamh@rs269poc-ls0.1@fg0.4-bg0.1-crf10-gt0.9@aff_fg=0.40_bg=0.10
# run_training
# run_inference

# TAG=rw/voc12-an@ccamh@rs269pnoc-ls0.1@fg0.3-bg0.1-crf10-gt0.9@aff_fg=0.30_bg=0.10
# CAMS_DIR=./experiments/predictions/pnoc/voc12-rs269-pnoc-ls0.1-ow0.0-1.0-1.0-cams-0.2-octis1-amp@rs269ra-r3@train@scale=0.5,1.0,1.5,2.0
# LABEL_DIR=./experiments/predictions/voc12-an@ccamh@rs269pnoc-ls0.1@fg0.3-bg0.1-crf10-gt0.9@aff_fg=0.30_bg=0.10
# run_training
# run_inference

# TAG=rw/voc12-an@rs269pnoc-ls0.1@fg0.3-bg0.1-crf10-gt0.7@aff_fg=0.30_bg=0.10
# CAMS_DIR=./experiments/predictions/pnoc/voc12-rs269-pnoc-ls0.1-ow0.0-1.0-1.0-cams-0.2-octis1-amp@rs269ra-r3@train@scale=0.5,1.0,1.5,2.0
# LABEL_DIR=./experiments/predictions/voc12-an@rs269pnoc-ls0.1@fg0.3-bg0.1-crf10-gt0.7@aff_fg=0.30_bg=0.10
# run_training
# run_inference

# TAG=rw/voc12-an@rs269pnoc-ls0.1@fg0.3-bg0.05-crf10-gt0.7@aff_fg=0.30_bg=0.05
# CAMS_DIR=./experiments/predictions/pnoc/voc12-rs269-pnoc-ls0.1-ow0.0-1.0-1.0-cams-0.2-octis1-amp@rs269ra-r3@train@scale=0.5,1.0,1.5,2.0
# LABEL_DIR=./experiments/predictions/voc12-an@rs269pnoc-ls0.1@fg0.3-bg0.05-crf10-gt0.7@aff_fg=0.30_bg=0.05
# run_training
# run_inference

# TAG=rw/voc12-an@ccamh@rs269pnoc-ls0.1@fg0.3-bg0.1-crf10-gt0.7
# CAMS_DIR=./experiments/predictions/pnoc/voc12-rs269-pnoc-ls0.1-ow0.0-1.0-1.0-cams-0.2-octis1-amp@rs269ra-r3@train@scale=0.5,1.0,1.5,2.0
# LABEL_DIR=./experiments/predictions/voc12-an@ccamh@rs269pnoc-ls0.1@crf10-gt0.7@aff_fg=0.30_bg=0.10
# run_training
# CAMS_DIR=./experiments/predictions/pnoc/voc12-rs269-pnoc-ls0.1-ow0.0-1.0-1.0-cams-0.2-octis1-amp@rs269ra-r3@train@scale=0.5,1.0,1.5,2.0
# DOMAIN=train_aug
# run_inference
# CAMS_DIR=./experiments/predictions/pnoc/voc12-rs269-pnoc-ls0.1-ow0.0-1.0-1.0-cams-0.2-octis1-amp@rs269ra-r3@val@scale=0.5,1.0,1.5,2.0
# DOMAIN=val
# run_inference


## =========================================
## MS COCO 14 Dataset
## =========================================

DATASET=coco14
DATA_DIR=/home/ldavid/workspace/datasets/coco14/
DOMAIN=train2014

TAG=rw/coco14-an@pnoc-ls0.1-ccamh-ls0.1@rs269ra
CAMS_DIR=pnoc/coco14-rs269-pnoc-b16-a2-ls0.1-ow0.0-1.0-1.0-c0.2-is1@rs269ra-r3@train@scale=0.5,1.0,1.5,2.0
LABEL_DIR=./experiments/predictions/affinity/coco14-rs269pnoc-ls@ccamh-rs269-fg0.2-ls@pn@an-crf10-gt0.7@aff_fg=0.30_bg=0.80
# run_training
run_inference
