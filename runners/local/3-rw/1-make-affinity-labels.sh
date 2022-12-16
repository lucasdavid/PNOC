

export PYTHONPATH=$(pwd)
export OMP_NUM_THREADS=8
export WANDB_PROJECT=research-wsss-dev

PY=python
SOURCE=scripts/rw/make_affinity_labels.py
DEVICE=cuda
WORKERS=8

# DATASET=voc12
# DATA_DIR=/home/ldavid/workspace/datasets/voc/VOCdevkit/VOC2012/
# DOMAIN=train_aug
DATASET=coco14
DATA_DIR=/home/ldavid/workspace/datasets/coco14/
DOMAIN=train2014


run_make_affinity_labels() {
    CUDA_VISIBLE_DEVICES=""   \
    $PY $SOURCE               \
        --tag     $TAG        \
        --dataset $DATASET    \
        --domain  $DOMAIN     \
        --fg_threshold $FG    \
        --bg_threshold $BG    \
        --crf_t $CRF_T        \
        --crf_gt_prob $CRF_GT \
        --cams_dir $CAMS_DIR  \
        --sal_dir  "$SAL_DIR"   \
        --data_dir $DATA_DIR  \
        --num_workers $WORKERS
}


# FG=0.4
# BG=0.1
# CRF_T=10
# CRF_GT=0.9
# CAMS_DIR=./experiments/predictions/poc/ResNeSt269@PuzzleOc@train@scale=0.5,1.0,1.5,2.0/
# SAL_DIR=./experiments/predictions/saliency/poolnet@ccam-fgh@rs269@rs269-poc/
# TAG=affnet@rs269-poc@pn-fgh@crf-$CRF_T-gt-$CRF_GT
# run_make_affinity_labels

# FG=0.5
# BG=0.05
# TAG=affnet@rs269-poc@pn-fgh@crf-$CRF_T-gt-$CRF_GT
# run_make_affinity_labels


FG=0.4
BG=0.1
CRF_T=10
CRF_GT=0.9
CAMS_DIR=./experiments/predictions/coco14/rs50@train@scale=0.5,1.0,1.5,2.0
TAG=coco14/affnet@rs50@crf-$CRF_T-gt-$CRF_GT
run_make_affinity_labels
