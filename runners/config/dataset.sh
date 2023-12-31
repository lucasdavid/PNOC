# Dataset

if [[ $DATASET == voc12 ]]; then
  DOMAIN_TRAIN=train_aug
  DOMAIN_VALID=train
  DOMAIN_VALID_SEG=val
  DOMAIN_TEST="test"
  DATA_DIR=$DATASETS_DIR/VOCdevkit/VOC2012

  IMAGE_SIZE=512
  MIN_IMAGE_SIZE=320
  MAX_IMAGE_SIZE=640

  VALIDATE_MAX_STEPS=0
  VALIDATE_THRESHOLDS=0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45

  LR=0.1
  WD=0.0001
fi
if [[ $DATASET == coco14 ]]; then
  DOMAIN_TRAIN=train2014
  DOMAIN_VALID=train2014
  DOMAIN_VALID_SEG=val2014
  DOMAIN_TEST=val2014
  DATA_DIR=$DATASETS_DIR/coco14

  IMAGE_SIZE=640
  MIN_IMAGE_SIZE=400
  MAX_IMAGE_SIZE=800

  VALIDATE_MAX_STEPS=620  # too many valid samples
  VALIDATE_THRESHOLDS=0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45

  LR=0.05
  WD=0.0001
fi
if [[ $DATASET == deepglobe ]]; then
  DOMAIN_TRAIN=train75
  DOMAIN_VALID=train75
  DOMAIN_VALID_SEG="test"
  DOMAIN_TEST="test"
  DATA_DIR=$DATASETS_DIR/DGdevkit

  IMAGE_SIZE=320
  MIN_IMAGE_SIZE=$IMAGE_SIZE
  MAX_IMAGE_SIZE=$IMAGE_SIZE

  VALIDATE_MAX_STEPS=0
  # A single threshold is used because the bg class is not added based on
  # thresholding. We assume bg=agriculture_land (the most frequent class).
  VALIDATE_THRESHOLDS=0.5

  LR=0.1
  WD=0.0001
fi
if [[ $DATASET == cityscapes ]]; then
  DOMAIN_TRAIN=train
  DOMAIN_VALID=train
  DOMAIN_VALID_SEG="val"
  DOMAIN_TEST="test"
  DATA_DIR=$DATASETS_DIR/cityscapes

  IMAGE_SIZE=768
  MIN_IMAGE_SIZE=768  # 960   # Images in this set have sizes (1024, 2048). Ratios are calculated over 2028,
  MAX_IMAGE_SIZE=768  # 1920  # hence min/max are larger than `IMAGE_SIZE`. True (min, max) = (480, 960).

  VALIDATE_MAX_STEPS=0
  VALIDATE_THRESHOLDS=0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45

  LR=0.1
  WD=0.0001
fi
