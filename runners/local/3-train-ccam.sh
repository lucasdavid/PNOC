
PY=python
SOURCE=train_ccam.py
DEVICE=cpu
WORKERS=8
DATA_DIR=/mnt/files/Workspace/datasets/voc/VOCdevkit/VOC2012/

IMAGE_SIZE=64
BATCH_SIZE=8

ARCHITECTURE=resnest50
DILATED=false
TRAINABLE_STEM=true
MODE=normal

TAG=ccam@$ARCHITECTURE

# pip install cmapy


$PY $SOURCE                          \
  --tag             $TAG             \
  --device          $DEVICE          \
  --batch_size      $BATCH_SIZE      \
  --num_workers     $WORKERS         \
  --architecture    $ARCHITECTURE    \
  --dilated         $DILATED         \
  --mode            $MODE            \
  --trainable-stem  $TRAINABLE_STEM  \
  --image_size      $IMAGE_SIZE      \
  --data_dir        $DATA_DIR
