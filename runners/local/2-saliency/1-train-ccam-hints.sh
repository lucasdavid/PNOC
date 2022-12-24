export PYTHONPATH=$(pwd)
export OMP_NUM_THREADS=8
export WANDB_PROJECT=research-wsss-dev

PY=python
SOURCE=scripts/ccam/train_with_cam_hints.py
DEVICE=cuda
DEVICES=0
WORKERS=8

DATASET=voc12
DATA_DIR=/home/ldavid/workspace/datasets/voc/VOCdevkit/VOC2012/
# DATASET=coco14
# DATA_DIR=/home/ldavid/workspace/datasets/coco14/

IMAGE_SIZE=64
BATCH_SIZE=8
EPOCHS=5
ACCUMULATE_STEPS=1
MIXED_PRECISION=true

ARCHITECTURE=resnet50
ARCH=rn50
DILATED=false
TRAINABLE_STEM=true
MODE=normal
S4_OUT_FEATURES=1024
WEIGHTS=./experiments/models/moco_r50_v2-e3b0c442.pth  # imagenet

ALPHA=0.25
LR=0.0001

TAG=ccam-$ARCH-moco-e$EPOCHS-b$BATCH_SIZE-lr$LR

WANDB_PROJECT=research-wsss-dev           \
WANDB_TAGS="$DATASET,$ARCH,ccam"          \
CUDA_VISIBLE_DEVICES=$DEVICES $PY $SOURCE \
  --tag             $TAG                  \
  --alpha           $ALPHA                \
  --max_epoch       $EPOCHS               \
  --batch_size      $BATCH_SIZE           \
  --lr              $LR                   \
  --accumule_steps  $ACCUMULATE_STEPS     \
  --mixed_precision $MIXED_PRECISION      \
  --num_workers     $WORKERS              \
  --architecture    $ARCHITECTURE         \
  --stage4_out_features $S4_OUT_FEATURES  \
  --dilated         $DILATED              \
  --mode            $MODE                 \
  --weights         $WEIGHTS              \
  --trainable-stem  $TRAINABLE_STEM       \
  --image_size      $IMAGE_SIZE           \
  --data_dir        $DATA_DIR
