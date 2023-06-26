

export PYTHONPATH=$(pwd)
# export OMP_NUM_THREADS=4

PY=python
SOURCE=scripts/rw/inference.py
DEVICE=cuda
DEVICES=0
WORKERS=8

MIXED_PRECISION=true  # false

# Dataset
DATASET=voc12
DATA_DIR=/home/ldavid/workspace/datasets/voc/VOCdevkit/VOC2012/
DOMAIN=train

IMAGE_SIZE=512
MIN_IMAGE_SIZE=320
MAX_IMAGE_SIZE=640

# Architecture
ARCHITECTURE=resnest269
ARCH=rs269
BATCH_SIZE=32
LR=0.1


rw_make_affinity_labels() {
    $PY scripts/rw/rw_make_affinity_labels.py \
    --tag          $AFF_LABELS_TAG \
    --dataset      $DATASET   \
    --domain       $DOMAIN    \
    --fg_threshold $FG        \
    --bg_threshold $BG        \
    --crf_t        $CRF_T     \
    --crf_gt_prob  $CRF_GT    \
    --cams_dir     $CAMS_DIR  \
    --sal_dir      $SAL_DIR   \
    --data_dir     $DATA_DIR  \
    --num_workers  $WORKERS
}

rw_training() {
    CUDA_VISIBLE_DEVICES=$DEVICES    \
    WANDB_TAGS="$DATASET,$ARCH,rw"   \
    $PY scripts/rw/train_affinity.py \
        --architecture $ARCHITECTURE \
        --tag          $AFF_TAG       \
        --batch_size   $BATCH_SIZE   \
        --image_size   $IMAGE_SIZE   \
        --min_image_size $MIN_IMAGE_SIZE \
        --max_image_size $MAX_IMAGE_SIZE \
        --dataset      $DATASET      \
        --lr           $LR           \
        --label_dir    $AFF_LABELS_DIR \
        --data_dir     $DATA_DIR     \
        --num_workers  $WORKERS
}

rw_inference() {
    CUDA_VISIBLE_DEVICES=$DEVICES    \
    $PY $SOURCE                      \
        --architecture $ARCHITECTURE \
        --image_size   $IMAGE_SIZE   \
        --model_name   $MODEL_NAME   \
        --cam_dir      $CAM_DIR      \
        --domain       $DOMAIN       \
        --beta         10            \
        --exp_times    8             \
        --mixed_precision $MIXED_PRECISION \
        --dataset      $DATASET      \
        --data_dir     $DATA_DIR
}

make_pseudo_labels() {
  $PY scripts/segmentation/make_pseudo_labels.py \
      --experiment_name $AFF_TAG   \
      --dataset         $DATASET   \
      --domain          $DOMAIN    \
      --threshold       $THRESHOLD \
      --crf_t           $CRF_T     \
      --crf_gt_prob     $CRF_GT    \
      --data_dir        $DATA_DIR
}

## 3.1 Make Affinity Labels
##
PRIORS_TAG=ra-oc-p-poc-pnoc-avg
# PRIORS_TAG=ra-oc-p-poc-pnoc-learned-a0.25
AFF_TAG=rw/$DATASET-an@ccamh@$PRIORS_TAG

CAMS_DIR=./experiments/predictions/ensemble/$PRIORS_TAG
SAL_DIR=./experiments/predictions/saliency/voc12-pn@ccamh-rs269@$PRIORS_TAG
FG=0.3
BG=0.1
CRF_T=10
CRF_GT=0.7

AFF_LABELS_TAG=rw/$DATASET-an@ccamh@$PRIORS_TAG@crf$CRF_T-gt$CRF_GT
AFF_LABELS_DIR=./experiments/predictions/$AFF_LABELS_TAG@aff_fg=$FG_bg=$BG
# rw_make_affinity_labels

## 3.2. Affinity Net Train
##
# rw_training

## 3.3. Affinity Net Inference
##
MODEL_NAME=rw/voc12-an@ccamh@ra-oc-p-poc-pnoc-avg
RW_DIR=$MODEL_NAME@$DOMAIN@beta=10@exp_times=8@rw

CAM_DIR=./experiments/predictions/ensemble/ra-oc-p-poc-pnoc-avg
DOMAIN=train && rw_inference
# DOMAIN=val && rw_inference
