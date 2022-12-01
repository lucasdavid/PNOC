
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$(pwd)
export OMP_NUM_THREADS=8

PY=python
SOURCE=scripts/cam/train_puzzle.py
DEVICE=cpu
WORKERS=8

# DATASET=voc12
# DATA_DIR=/home/ldavid/workspace/datasets/voc/VOCdevkit/VOC2012/
DATASET=coco14
DATA_DIR=/home/ldavid/workspace/datasets/coco14/

IMAGE_SIZE=384
AUGMENT=mixup
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

TAG=$DATASET-$ARCH-p-aug_$AUGMENT


$PY $SOURCE                          \
  --tag             $TAG             \
  --num_workers     $WORKERS         \
  --batch_size      $BATCH           \
  --architecture    $ARCHITECTURE    \
  --dilated         $DILATED         \
  --mode            $MODE            \
  --trainable-stem  $TRAINABLE_STEM  \
  --mode            normal           \
  --re_loss_option  masking          \
  --re_loss         L1_Loss          \
  --alpha_schedule  0.50             \
  --alpha           4.00             \
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
