

export PYTHONPATH=$(pwd)
export OMP_NUM_THREADS=8
export WANDB_PROJECT=research-wsss-dev

PY=python
SOURCE=scripts/cam/train_apoc.py
DEVICE=cpu
WORKERS=4
AMP=true

DATASET=voc12
DATA_DIR=/home/ldavid/workspace/datasets/voc/VOCdevkit/VOC2012/
# DATASET=coco14
# DATA_DIR=/home/ldavid/workspace/datasets/coco14/

IMAGE_SIZE=64
AUGMENT=colorjitter
CUTMIX=0.5
MIXUP=1.
LABELSMOOTHING=0.1

EPOCHS=3
BATCH=8

ARCHITECTURE=resnet50
ARCH=rn50

DILATED=false
MODE=normal
TRAINABLE_STEM=true

OC_ARCHITECTURE=resnet50
OC_PRETRAINED=./experiments/models/ResNet50.pth
# OC_PRETRAINED=./experiments/models/coco14-rs50.pth

TAG=$DATASET-$ARCH-apoc@rn50


$PY $SOURCE                          \
  --tag             $TAG             \
  --device          $DEVICE          \
  --num_workers     $WORKERS         \
  --mixed_precision $AMP             \
  --batch_size      $BATCH           \
  --architecture    $ARCHITECTURE    \
  --dilated         $DILATED         \
  --mode            $MODE            \
  --trainable-stem  $TRAINABLE_STEM  \
  --alpha_schedule  0.50             \
  --alpha           4.00             \
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
