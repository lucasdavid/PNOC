# Research Weakly Supervised

Studying regularization strategies for WSSS.
Experiments were run over LNCC SDumont infrastructure.

Many of the code lines here were borrowed from OC-CSE, Puzzle-CAM and CCAM repositories.

## Setup
### Pascal VOC 2012

```shell
DATA_DIR=/datasets
cd $DATA_DIR

# Download images and labels from http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit:
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xf VOCtrainval_11-May-2012.tar

# Download the augmented segmentation maps from http://home.bharathh.info/pubs/codes/SBD
# *only if* you are planning on training the fully-suppervised
# models (for comparison purposes):
wget http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz
tar -xzf benchmark.tgz

# Merge them into the VOCdevkit/VOC2012/SegmentationClass folder:
mv aug_seg/* VOCdevkit/VOC2012/SegmentationClass/
```
### MS COCO 2014

```shell
DATA_DIR=/datasets/coco14
cd $DATA_DIR

# Download MS COCO images and labels:
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
wget https://github.com/jbeomlee93/RIB/raw/main/coco14/cls_labels.npy
wget https://github.com/jbeomlee93/RIB/raw/main/coco14/cls_labels_coco.npy

# Download segmentation labels in VOC format from
# https://drive.google.com/file/d/1pRE9SEYkZKVg0Rgz2pi9tg48j7GlinPV/view
gdown 1pRE9SEYkZKVg0Rgz2pi9tg48j7GlinPV

# Unzip everything:
unzip train2014.zip
unzip val2014.zip
unzip coco_annotations_semantic.zip
mv coco_annotations_semantic/coco_seg_anno .
rm coco_annotations_semantic -r
```

## Experiments

### 0. Common Setup
```shell
export PYTHONPATH=$(pwd)

PY=python3.9
PIP=pip3.9
DEVICES=0,1,2,3
WORKERS=24

$PIP install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu111
$PIP install -r requirements.txt

# WANDB_PROJECT=some-project-id  # Specify project to export training reports
# wandb disabled                 # Otherwise, no metrics exported.

DATASET=voc12
D_TRAIN=train_aug
D_VAL=val
DATA_DIR=/datasets/VOCdevkit/VOC2012
# DATASET=coco14
# D_TRAIN=train2014
# D_VAL=val2014
# DATA_DIR=/datasets/coco14

ARCH=resnest269
B=32    # batch size
AC=1    # accumulate steps
LS=0.1  # label smoothing
```

### 1. Train classifiers and generate segmentation priors
```shell
# 1.1 Train Ordinary Classifier (OC).
OC_TAG=$DATASET-rs269ra
OC_WEIGHTS=./experiments/models/$OC_TAG.pth

CUDA_VISIBLE_DEVICES=$DEVICES    \
$PY scripts/cam/train_vanilla.py \
  --tag             $OC_TAG      \
  --batch_size      $B           \
  --architecture    $ARCH        \
  --augment         randaugment  \
  --label_smoothing $LS          \
  --dataset         $DATASET     \
  --data_dir        $DATA_DIR

# 1.2 Train Puzzle-Not So Ordinary Classifier (P-NOC).
PNOC_TAG=$DATASET-rs269pnoc@rs269ra
PNOC_CAMS=$PNOC_TAG@$D_TRAIN@scale=0.5,1.0,1.5,2.0

CUDA_VISIBLE_DEVICES=$DEVICES     \
$PY scripts/cam/train_pnoc.py     \
  --tag               $PNOC_TAG   \
  --batch_size        $B          \
  --accumulate_steps  $AC         \
  --mixed_precision   true        \
  --augment           colorjitter \
  --label_smoothing   $LS         \
  --architecture      $ARCH       \
  --oc-architecture   $ARCH       \
  --oc-train-masks    cams        \
  --oc_train_mask_t   0.2         \
  --oc-pretrained     $OC_WEIGHTS \
  --dataset           $DATASET    \
  --data_dir          $DATA_DIR

# 1.3 Inference of CAMs with TTA.
CUDA_VISIBLE_DEVICES=$DEVICES $PY scripts/cam/inference.py --domain $D_TRAIN   \
  --architecture $ARCH --tag $PNOC_TAG --dataset $DATASET --data_dir $DATA_DIR
CUDA_VISIBLE_DEVICES=$DEVICES $PY scripts/cam/inference.py --domain $D_VAL     \
  --architecture $ARCH --tag $PNOC_TAG --dataset $DATASET --data_dir $DATA_DIR

# 1.4 (Optional) Evaluate CAM priors.
$PY scripts/evaluate.py --experiment_name $PNOC_CAMS --domain $D_TRAIN \
  --dataset $DATASET --data_dir $DATA_DIR --num_workers $WORKERS
```

### 2 Train CCAM and generate saliency hints
```shell
# 2.1 Train C²AM-H
FG_T=0.3  # Might need fine-tuning. Usually a high value, inducing low FP for classes.
CCAMH_TAG=$DATASET-ccamh-rs269-fg$FG_T@rs269pnoc@rs269ra

B=128
AC=1

CUDA_VISIBLE_DEVICES=$DEVICES       \
  $PY scripts/ccam/train_hints.py   \
  --tag                 $CCAMH_TAG  \
  --batch_size          $B          \
  --accumulate_steps    $AC         \
  --mixed_precision     true        \
  --architecture        $ARCH       \
  --stage4_out_features 1024        \
  --fg_threshold        $FG_T       \
  --cams_dir ./experiments/predictions/$PNOC_CAMS  \
  --dataset             $DATASET    \
  --data_dir            $DATA_DIR

# 2.2 Infer C²AM-H's saliency priorswith TTA:
CUDA_VISIBLE_DEVICES=$DEVICES $PY scripts/ccam/inference.py --tag $CCAMH_TAG \
  --architecture $ARCH --stage4_out_features 1024                            \
  --domain $D_TRAIN --dataset $DATASET --data_dir $DATA_DIR

CUDA_VISIBLE_DEVICES="" $PY scripts/ccam/inference_crf.py --experiment_name $CCAMH_TAG \
  --domain $D_TRAIN --dataset $DATASET --data_dir $DATA_DIR

# 2.3 Train PoolNet
PN_TAG=$DATASET-pn@ccamh@pnoc@ra
cd poolnet
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 $PY main_voc.py --arch resnet --mode train --train_root $DATA_DIR --pseudo_root ../experiments/predictions/$CCAMH_TAG
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 $PY main_voc.py --arch resnet --mode test --train_root $DATA_DIR --model ./results/path/to/your/saved/model --sal_folder ../experiments/predictions/$PN_TAG
cd ..
```

#### 3 Refining Priors with Random Walk
```shell
# 3.1 Creating Affinity Labels
FG=0.3
BG=0.1
CRF_T=10
CRF_GT=0.7
B=32
RW_TAG=$DATASET-an-fg$FG-bg$BG@pn@ccamh@pnoc@ra

CUDA_VISIBLE_DEVICES=""                 \
$PY scripts/rw/make_affinity_labels.py  \
    --tag          $RW_TAG              \
    --fg_threshold $FG                  \
    --bg_threshold $BG                  \
    --crf_t        $CRF_T               \
    --crf_gt_prob  $CRF_GT              \
    --cams_dir ./experiments/predictions/$PNOC_CAMS \
    --sal_dir  ./experiments/predictions/$PN_TAG    \
    --num_workers  $WORKERS             \
    --domain       $D_TRAIN             \
    --dataset      $DATASET             \
    --data_dir     $DATA_DIR

# 3.2 Train AffinityNet
CUDA_VISIBLE_DEVICES=$DEVICES    \
$PY scripts/rw/train_affinity.py \
    --tag          $RW_TAG       \
    --architecture $ARCH         \
    --batch_size   $B            \
    --label_dir    ./experiments/predictions/$RW_TAG  \
    --dataset      $DATASET      \
    --data_dir     $DATA_DIR

# 3.3 Affinity Random Walk
CUDA_VISIBLE_DEVICES=$DEVICES $PY scripts/rw/inference.py --domain $D_TRAIN \
  --architecture $ARCH --model_name $RW_TAG --cam_dir $PNOC_TAG             \
  --beta 10 --exp_times 8 --dataset $DATASET --data_dir $DATA_DIR

CUDA_VISIBLE_DEVICES=$DEVICES $PY scripts/rw/inference.py --domain $D_VAL \
  --architecture $ARCH --model_name $RW_TAG --cam_dir $PNOC_TAG           \
  --beta 10 --exp_times 8 --dataset $DATASET --data_dir $DATA_DIR

# 3.4 Generate pseudo labels
T=0.3
CRF_T=1
CRF_P=0.9
RW_LABELS=$RW_TAG@beta=10@exp_times=8@rw@crf=$CRF_T

CUDA_VISIBLE_DEVICES="" $PY scripts/segmentation/make_pseudo_labels.py     \
  --experiment_name $RW_TAG@train@beta=10@exp_times=8@rw --domain $D_TRAIN \
  --threshold $T --crf_t $CRF_T --crf_gt_prob $CRF_P                       \
  --dataset $DATASET --data_dir $DATA_DIR                                  &
CUDA_VISIBLE_DEVICES="" $PY scripts/segmentation/make_pseudo_labels.py     \
  --experiment_name $RW_TAG@val@beta=10@exp_times=8@rw --domain $D_VAL     \
  --threshold $T --crf_t $CRF_T --crf_gt_prob $CRF_P                       \
  --dataset $DATASET --data_dir $DATA_DIR                                  &
wait

# Merge train and val pseudo labels into a single folder:
mkdir -p $RW_LABELS
mv $RW_TAG@train@beta=10@exp_times=8@rw@crf=1/* $RW_LABELS/
mv $RW_TAG@val@beta=10@exp_times=8@rw@crf=1/* $RW_LABELS/
rm $RW_TAG@train@beta=10@exp_times=8@rw@crf=1 $RW_TAG@val@beta=10@exp_times=8@rw@crf=1 -r
```

#### 4 Semantic Segmentation
```shell
# 4.2 Train DeepLabV3+
SEGM_TAG=dlv3p-gn@an@pn@ccamh@rs269pnoc@rs269ra
B=32
AUG=colorjitter  # colorjitter_cutmix

CUDA_VISIBLE_DEVICES=$DEVICES     \
$PY scripts/segmentation/train.py \
    --tag             $SEGM_TAG   \
    --backbone        $ARCH       \
    --batch_size      $B          \
    --use_gn          true        \
    --mixed_precision true        \
    --augment         $AUG        \
    --label_smoothing $LS         \
    --label_name      $RW_LABELS  \
    --dataset         $DATASET    \
    --data_dir        $DATA_DIR

# 4.3 Inference with DeepLabV3+ and TTA:
DOMAIN=val
CUDA_VISIBLE_DEVICES=$DEVICES                                \
$PY scripts/segmentation/inference.py --tag $SEGM_TAG        \
  --backbone $ARCH --use_gn true --crf_t 1 --crf_gt_prob 0.9 \
  --dataset $DATASET --domain $DOMAIN --data_dir $DATA_DIR
```


## Results
### Pascal VOC 2012 (test)

| bg | a.plane | bike | bird  | boat  | bottle | bus   | car   | cat   | chair | cow   | d.table | dog   | horse | m.bike | person | p.plant | sheep | sofa  | train | tv | Overall |
| ---------- | --------- | ------- | ----- | ----- | ------ | ----- | ----- | ----- | ----- | ----- | ----------- | ----- | ----- | --------- | ------ | ----------- | ----- | ----- | ----- | --------- | ------- |
| 91.55      | 86.74     | 38.28   | 89.29 | 61.13 | 74.81  | 92.01 | 86.57 | 89.91 | 20.53 | 85.81 | 56.98       | 90.21 | 83.53 | 83.38     | 80.78  | 67.99       | 86.96 | 47.09 | 62.76 | 43.09     | 72.35   |
| 91.36      | 86.70     | 35.18   | 87.84 | 62.89 | 71.57  | 92.97 | 86.33 | 92.34 | 30.43 | 85.79 | 60.68       | 91.73 | 81.70 | 82.72     | 66.30  | 65.85       | 88.75 | 48.71 | 72.48 | 44.48     | 72.70   |
