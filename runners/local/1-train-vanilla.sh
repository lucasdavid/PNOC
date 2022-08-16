
PY=python
SOURCE=train_classification.py
DEVICE=cpu
WORKERS=8
DATA_DIR=/mnt/files/Workspace/datasets/voc/VOCdevkit/VOC2012/

IMAGE_SIZE=64

ARCHITECTURE=res2net101_v1b
DILATED=false
TRAINABLE_STEM=true
MODE=normal


$PY $SOURCE                          \
  --device          $DEVICE          \
  --num_workers     $WORKERS         \
  --architecture    $ARCHITECTURE    \
  --dilated         $DILATED         \
  --mode            $MODE            \
  --trainable-stem  $TRAINABLE_STEM  \
  --image_size      $IMAGE_SIZE      \
  --data_dir        $DATA_DIR
