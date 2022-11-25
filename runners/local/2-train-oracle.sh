
PY=python
SOURCE=train_occse_oracle.py
DEVICE=cpu
WORKERS=8
DATA_DIR=/home/ldavid/workspace/datasets/voc/VOCdevkit/VOC2012/

IMAGE_SIZE=64

ARCHITECTURE=resnet50
DILATED=false
TRAINABLE_STEM=true
MODE=normal
# restore=/path/to/restore

OC_ARCHITECTURE=resnet50
OC_PRETRAINED=./experiments/models/ResNet50.pth


$PY $SOURCE                          \
  --device          $DEVICE          \
  --num_workers     $WORKERS         \
  --architecture    $ARCHITECTURE    \
  --dilated         $DILATED         \
  --mode            $MODE            \
  --trainable-stem  $TRAINABLE_STEM  \
  --oc-architecture $OC_ARCHITECTURE \
  --oc-pretrained   $OC_PRETRAINED   \
  --image_size      $IMAGE_SIZE      \
  --data_dir        $DATA_DIR
