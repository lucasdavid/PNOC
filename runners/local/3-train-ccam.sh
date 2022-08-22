
PY=python
SOURCE=ccam_train_with_cam_hints.py
DEVICE=cpu
WORKERS=8
DATA_DIR=/mnt/files/Workspace/datasets/voc/VOCdevkit/VOC2012/
IMAGE_SIZE=64
BATCH_SIZE=8

ARCHITECTURE=resnet38d
DILATED=false
TRAINABLE_STEM=true
MODE=normal

TAG=ccam@$ARCHITECTURE

CAMS_DIR=experiments/predictions/resnest101@randaug@train@scale=0.5,1.0,1.5,2.0


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
  --cams_dir        $CAMS_DIR        \
  --data_dir        $DATA_DIR
