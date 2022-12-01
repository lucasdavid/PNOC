
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$(pwd)
export OMP_NUM_THREADS=8

PY=python
SOURCE=scripts/cam/train_occse.py
DEVICE=cpu
WORKERS=8

# DATASET=voc12
# DATA_DIR=/home/ldavid/workspace/datasets/voc/VOCdevkit/VOC2012/
DATASET=coco14
DATA_DIR=/home/ldavid/workspace/datasets/coco14/

IMAGE_SIZE=384
AUGMENT=none
CUTMIX=0.5
MIXUP=1.
LABELSMOOTHING=0.1

EPOCHS=15
BATCH=8

ARCHITECTURE=resnest50
ARCH=rs50

DILATED=false
MODE=normal
TRAINABLE_STEM=true

OC_ARCHITECTURE=resnest50
OC_PRETRAINED=./experiments/models/coco14-rs50.pth
# OC_PRETRAINED=./experiments/models/voc12-rs50-ra_cm@0.5.pth

TAG=$DATASET-$ARCH-oc@rs50


$PY $SOURCE                          \
  --tag             $TAG             \
  --device          $DEVICE          \
  --num_workers     $WORKERS         \
  --batch_size      $BATCH           \
  --architecture    $ARCHITECTURE    \
  --dilated         $DILATED         \
  --mode            $MODE            \
  --trainable-stem  $TRAINABLE_STEM  \
  --oc-architecture $OC_ARCHITECTURE \
  --oc-pretrained   $OC_PRETRAINED   \
  --image_size      $IMAGE_SIZE      \
  --min_image_size  $IMAGE_SIZE      \
  --max_image_size  $IMAGE_SIZE      \
  --augment         $AUGMENT         \
  --cutmix_prob     $CUTMIX          \
  --mixup_prob      $MIXUP           \
  --label_smoothing $LABELSMOOTHING  \
  --max_epoch       $EPOCHS          \
  --dataset         $DATASET         \
  --data_dir        $DATA_DIR
