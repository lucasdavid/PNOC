

export PYTHONPATH=$(pwd)

PY=python
SOURCE=scripts/cam/train_poc.py
DEVICE=cuda
DEVICES=0
WORKERS=8

DATASET=voc12
DATA_DIR=/home/ldavid/workspace/datasets/voc/VOCdevkit/VOC2012/
DOMAIN=train
# DATASET=coco14
# DATA_DIR=/home/ldavid/workspace/datasets/coco14/
# DOMAIN=train2014


ARCH=rn50
ARCHITECTURE=resnet50
TRAINABLE_STEM=true
DILATED=false
MODE=normal
REGULAR=none

IMAGE_SIZE=320
MIN_IMAGE_SIZE=260
MAX_IMAGE_SIZE=380
EPOCHS=5
BATCH=8
ACCUMULATE_STEPS=2

OC_NAME=rn50
OC_PRETRAINED=experiments/models/ResNet50.pth
OC_ARCHITECTURE=$ARCHITECTURE
OC_REGULAR=none

OC_STRATEGY=random
OC_MASK_GN=true
OC_F_MOMENTUM=0.8
OC_F_GAMMA=5.0
# Schedule
P_INIT=0.0
P_ALPHA=4.0
P_SCHEDULE=0.5
OC_INIT=0.3
OC_ALPHA=1.0
OC_SCHEDULE=1.0

AUGMENT=colorjitter
CUTMIX=0.5
MIXUP=1.
LABELSMOOTHING=0  # 0.1

run_training () {
    CUDA_VISIBLE_DEVICES=$DEVICES            \
    $PY $SOURCE                              \
        --tag               $TAG             \
        --num_workers       $WORKERS         \
        --batch_size        $BATCH           \
        --accumulate_steps  $ACCUMULATE_STEPS \
        --architecture      $ARCHITECTURE    \
        --dilated           $DILATED         \
        --mode              $MODE            \
        --trainable-stem    $TRAINABLE_STEM  \
        --regularization    $REGULAR         \
        --oc-architecture   $OC_ARCHITECTURE \
        --oc-pretrained     $OC_PRETRAINED   \
        --oc-regularization $OC_REGULAR      \
        --oc-mask-globalnorm $OC_MASK_GN     \
        --image_size        $IMAGE_SIZE      \
        --min_image_size    $MIN_IMAGE_SIZE  \
        --max_image_size    $MAX_IMAGE_SIZE  \
        --augment           $AUGMENT         \
        --cutmix_prob       $CUTMIX          \
        --mixup_prob        $MIXUP           \
        --label_smoothing   $LABELSMOOTHING  \
        --max_epoch         $EPOCHS          \
        --alpha             $P_ALPHA         \
        --alpha_init        $P_INIT          \
        --alpha_schedule    $P_SCHEDULE      \
        --oc-alpha          $OC_ALPHA        \
        --oc-alpha-init     $OC_INIT         \
        --oc-alpha-schedule $OC_SCHEDULE     \
        --oc-strategy       $OC_STRATEGY     \
        --oc-focal-momentum $OC_F_MOMENTUM   \
        --oc-focal-gamma    $OC_F_GAMMA      \
        --dataset           $DATASET         \
        --data_dir          $DATA_DIR
}

run_inference () {
    CUDA_VISIBLE_DEVICES=$DEVICES        \
    $PY scripts/cam/inference.py         \
    --architecture      $ARCHITECTURE    \
    --regularization    $REGULAR         \
    --dilated           $DILATED         \
    --trainable-stem    $TRAINABLE_STEM  \
    --mode              $MODE            \
    --tag               $TAG             \
    --domain            $DOMAIN          \
    --data_dir          $DATA_DIR        &
}

LABELSMOOTHING=0.1
TAG=$DATASET-$ARCH-poc-ls$LABELSMOOTHING@$OC_NAME
run_training

# OC_MASK_GN=true
# OC_NAME=rn50
# OC_PRETRAINED=experiments/models/ResNet50.pth
# OC_ARCHITECTURE=resnet50
# DEVICES=0
# TAG=$DATASET-$ARCH-poc-ls$LABELSMOOTHING@$OC_NAME
# run_inference
