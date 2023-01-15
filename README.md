# Research Weakly Supervised

Studying regularization strategies for WSSS.
Experiments were run over LNCC SDumont infrastructure.

Many of the code lines here were borrowed from OC-CSE, Puzzle-CAM and CCAM repositories.

## Experiments

### Common Setup
```shell
export PYTHONPATH=$(pwd)

PY=python3.9
PIP=pip3.9
DEVICES=0,1,2,3

$PIP install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu111
$PIP install -r requirements.txt

# WANDB_PROJECT=some-project-id  # Specify project to export training reports
# wandb disabled                 # Otherwise, no metrics exported.

DATASET=voc12
DOMAIN_TRAIN=train_aug
DOMAIN_VALID=val
DATA_DIR=/datasets/VOCdevkit/VOC2012
# DATASET=coco14
# DOMAIN_TRAIN=train2014
# DOMAIN_VALID=val2014
# DATA_DIR=/datasets/coco14
```

### 1 Train classifiers and generate segmentation priors
#### 1.1 Train Ordinary Classifier (OC)
```shell
TAG_OC=$DATASET-rs269ra

CUDA_VISIBLE_DEVICES=$DEVICES        \
  $PY scripts/cam/train_vanilla.py   \
    --tag             $TAG_OC        \
    --batch_size      32             \
    --architecture    resnest269     \
    --augment         randaugment    \
    --label_smoothing 0.1            \
    --dataset         $DATASET       \
    --data_dir        $DATA_DIR
```

#### 1.2 Train Adversarial Puzzle-OC (APOC)
```shell
TAG_APOC=$DATASET-rs269apoc@rs269ra

CUDA_VISIBLE_DEVICES=$DEVICES            \
    python scripts/cam/train_apoc.py     \
        --tag               $TAG         \
        --batch_size        32           \
        --mixed_precision   true         \
        --architecture      resnest269   \
        --oc-architecture   resnest269   \
        --oc-pretrained     ./experiments/models/$TAG_OC  \
        --augment           colorjitter  \
        --label_smoothing   0.1          \
        --oc-train-masks    cams         \
        --oc_train_mask_t   0.2          \
        --dataset           $DATASET     \
        --data_dir          $DATA_DIR
```

#### 1.3 Inference of CAMs with TTA
```shell
CUDA_VISIBLE_DEVICES=0 $PY scripts/cam/inference.py --domain $DOMAIN_TRAIN --architecture resnest269 --tag $TAG_APOC --dataset $DATASET --data_dir $DATA_DIR &
CUDA_VISIBLE_DEVICES=1 $PY scripts/cam/inference.py --domain $DOMAIN_VALID --architecture resnest269 --tag $TAG_APOC --dataset $DATASET --data_dir $DATA_DIR &
wait
```

### 2 Train CCAM and generate saliency hints
#### 2.1 Train CCAM with hints
```shell
FG_T=0.4
TAG_CCAMH=$DATASET-ccamh-rs269@rs269apoc@rs269ra@$FG_T

CUDA_VISIBLE_DEVICES=$DEVICES               \
  $PY scripts/ccam/train_with_cam_hints.py  \
  --tag             $TAG_CCAMH              \
  --batch_size      128                     \
  --accumulate_steps  1                       \
  --mixed_precision true                    \
  --architecture    resnest269              \
  --stage4_out_features 1024                \
  --fg_threshold    $FG_T                   \
  --cams_dir ./experiments/predictions/$TAG_APOC  \
  --dataset         $DATASET                \
  --data_dir        $DATA_DIR
```

#### 2.2 Inference with TTA
```shell
CUDA_VISIBLE_DEVICES=0 $PY scripts/ccam/inference.py --tag $TAG_CCAMH --architecture resnest269 --stage4_out_features 1024 --domain train_aug --dataset $DATASET --data_dir $DATA_DIR

CUDA_VISIBLE_DEVICES="" $PY scripts/ccam/inference_crf.py --experiment_name $TAG_CCAMH --domain train_aug --dataset $DATASET --data_dir $DATA_DIR
```

#### 2.3 Train PoolNet
```shell
TAG_PN=$DATASET-pn@ccamh@rs269apoc@rs269ra

cd poolnet

OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 $PY main_voc.py --arch resnet --mode train --train_root $DATA_DIR --pseudo_root ../experiments/predictions/$TAG_CCAMH

OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 $PY main_voc.py --arch resnet --mode test --train_root $DATA_DIR --model ./results/path/to/your/saved/model --sal_folder ../experiments/predictions/$TAG_PN

cd ..
```

### 3 Refining Priors with Random Walk
#### 3.1 Creating Affinity Labels
```shell
FG=0.4
BG=0.1
CRF_T=10
CRF_GT=0.9
TAG_RW=$DATASET-an@pn@ccamh@rs269apoc@rs269ra@fg$FG-bg$BG-crf$CRF_T-gt$CRF_GT

CUDA_VISIBLE_DEVICES=""                 \
$PY scripts/rw/make_affinity_labels.py  \
    --tag          $TAG_RW              \
    --fg_threshold $FG                  \
    --bg_threshold $BG                  \
    --crf_t        $CRF_T               \
    --crf_gt_prob  $CRF_GT              \
    --cams_dir ./experiments/predictions/$TAG_APOC  \
    --sal_dir  ./experiments/predictions/$TAG_PN    \
    --num_workers  48                   \
    --domain       $DOMAIN_TRAIN        \
    --dataset      $DATASET             \
    --data_dir     $DATA_DIR
```

#### 3.2 Train AffinityNet
```shell
CUDA_VISIBLE_DEVICES=$DEVICES    \
$PY scripts/rw/train_affinity.py \
    --tag          $TAG_RW       \
    --architecture resnest269    \
    --batch_size   32            \
    --label_dir    ./experiments/predictions/$TAG_RW  \
    --dataset      $DATASET      \
    --data_dir     $DATA_DIR
```

#### 3.3 Affinity Random Walk
```shell
CUDA_VISIBLE_DEVICES=0 $PY scripts/rw/inference.py --domain $DOMAIN_TRAIN --architecture resnest269 --model_name $TAG_RW --cam_dir $TAG_APOC --beta 10 --exp_times 8 --dataset $DATASET --data_dir $DATA_DIR &

CUDA_VISIBLE_DEVICES=1 $PY scripts/rw/inference.py --domain $DOMAIN_VALID --architecture resnest269 --model_name $TAG_RW --cam_dir $TAG_APOC --beta 10 --exp_times 8 --dataset $DATASET --data_dir $DATA_DIR &

wait
```

### 4 Semantic Segmentation
#### 4.1 Generate pseudo labels
```shell
CUDA_VISIBLE_DEVICES="" $PY scripts/segmentation/make_pseudo_labels.py --experiment_name $TAG_RW@train@beta=10@exp_times=8@rw --domain $DOMAIN_TRAIN --threshold 0.3 --crf_t 1 --crf_gt_prob 0.9 --dataset $DATASET --data_dir $DATA_DIR &
CUDA_VISIBLE_DEVICES="" $PY scripts/segmentation/make_pseudo_labels.py --experiment_name $TAG_RW@val@beta=10@exp_times=8@rw --domain $DOMAIN_VALID --threshold 0.3 --crf_t 1 --crf_gt_prob 0.9 --dataset $DATASET --data_dir $DATA_DIR &

wait
```

#### 4.2 Train DeepLabV3+
```shell
TAG_DL=dlv3p-gn@an@pn@ccamh@rs269apoc@rs269ra

CUDA_VISIBLE_DEVICES=$DEVICES      \
$PY scripts/segmentation/train.py  \
    --backbone   resnest269        \
    --use_gn     true              \
    --tag        $TAG_DL           \
    --label_name $TAG_RW@val@beta=10@exp_times=8@rw@crf=1  \
    --dataset    $DATASET          \
    --data_dir   $DATA_DIR
```

#### 4.3 Inference with DeepLabV3+ and TTA
```shell

CUDA_VISIBLE_DEVICES=0 $PY scripts/segmentation/inference.py --domain train --backbone resnest269 --use_gn true --tag $TAG_DL --scale 0.5,1.0,1.5,2.0 --iteration 10 --data_dir $DATA_DIR &

CUDA_VISIBLE_DEVICES=1 $PY scripts/segmentation/inference.py --domain val --backbone resnest269 --use_gn true --tag $TAG_DL --scale 0.5,1.0,1.5,2.0 --iteration 10 --data_dir $DATA_DIR &

CUDA_VISIBLE_DEVICES=2 $PY scripts/segmentation/inference.py --domain test --backbone resnest269 --use_gn true --tag $TAG_DL --scale 0.5,1.0,1.5,2.0 --iteration 10 --data_dir $DATA_DIR &

wait
```
