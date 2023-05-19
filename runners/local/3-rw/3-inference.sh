

export PYTHONPATH=$(pwd)
# export OMP_NUM_THREADS=4

PY=python
SOURCE=scripts/rw/inference.py
DEVICE=cuda
DEVICES=0
WORKERS=8

MIXED_PRECISION=true  # false

# Dataset
IMAGE_SIZE=512

run_inference() {
    CUDA_VISIBLE_DEVICES=$DEVICES    \
    $PY $SOURCE                      \
        --architecture $ARCHITECTURE \
        --image_size   $IMAGE_SIZE   \
        --model_name   $MODEL_NAME   \
        --cam_dir      $CAM_DIR      \
        --domain       $DOMAIN       \
        --beta         10            \
        --exp_times    8             \
        --mixed_precision $MIXED_PRECISION \
        --dataset      $DATASET      \
        --data_dir     $DATA_DIR
}

run_crf() {
  $PY scripts/segmentation/make_pseudo_labels.py \
      --experiment_name $TAG       \
      --dataset         $DATASET   \
      --domain          $DOMAIN    \
      --threshold       $THRESHOLD \
      --crf_t           $CRF_T     \
      --crf_gt_prob     $CRF_GT    \
      --data_dir        $DATA_DIR
}

DATASET=voc12
DATA_DIR=/home/ldavid/workspace/datasets/voc/VOCdevkit/VOC2012/
DOMAIN=train

ARCHITECTURE=resnest269
# MODEL_NAME=affnet@rs269-poc@pn-fgh@crf-10-gt-0.9@aff_fg=0.40_bg=0.10
# CAM_DIR=poc/ResNeSt269@PuzzleOc@train@scale=0.5,1.0,1.5,2.0
# RW_DIR=rw/$MODEL_NAME@train@beta=10@exp_times=8@rw
# run_inference

DOMAIN=val

MODEL_NAME=rw/AffinityNet@ResNeSt-269@Puzzle
CAM_DIR=puzzle/ResNeSt269@Puzzle@optimal@$DOMAIN@scale=0.5,1.0,1.5,2.0
RW_DIR=$MODEL_NAME@$DOMAIN@beta=10@exp_times=8@rw
# run_inference

MODEL_NAME=rw/AffinityNet@resnest269@puzzlerep@aff_fg=0.40_bg=0.10
CAM_DIR=puzzle/resnest269@puzzlerep@$DOMAIN@scale=0.5,1.0,1.5,2.0
RW_DIR=$MODEL_NAME@$DOMAIN@beta=10@exp_times=8@rw
# run_inference

MODEL_NAME=rw/AffinityNet@ResNeSt269@PuzzleOc
CAM_DIR=poc/ResNeSt269@PuzzleOc@$DOMAIN@scale=0.5,1.0,1.5,2.0
RW_DIR=$MODEL_NAME@$DOMAIN@beta=10@exp_times=8@rw
# run_inference

MODEL_NAME=rw/voc12-an@rs269poc-ls0.1@fg0.4-bg0.1-crf10-gt0.9@aff_fg=0.40_bg=0.10
CAM_DIR=poc/voc12-rs269-poc-ls0.1@rs269ra-r3@$DOMAIN@scale=0.5,1.0,1.5,2.0
RW_DIR=$MODEL_NAME@$DOMAIN@beta=10@exp_times=8@rw
# run_inference
# $PY scripts/evaluate.py --experiment_name $RW_DIR --dataset voc12 --domain val --data_dir $DATA_DIR --min_th 0.05 --max_th 0.81 --crf_t 1 --crf_gt_prob 0.9

MODEL_NAME=rw/voc12-an@ccamh@rs269poc-ls0.1@fg0.4-bg0.1-crf10-gt0.9@aff_fg=0.40_bg=0.10
CAM_DIR=poc/voc12-rs269-poc-ls0.1@rs269ra-r3@$DOMAIN@scale=0.5,1.0,1.5,2.0
RW_DIR=$MODEL_NAME@$DOMAIN@beta=10@exp_times=8@rw
# run_inference
# $PY scripts/evaluate.py --experiment_name $RW_DIR --dataset voc12 --domain val --data_dir $DATA_DIR --min_th 0.05 --max_th 0.81 --crf_t 1 --crf_gt_prob 0.9

MODEL_NAME=rw/voc12-an@rs269pnoc-ls0.1@fg0.3-bg0.1-crf10-gt0.7@aff_fg=0.30_bg=0.10
CAM_DIR=pnoc/voc12-rs269-pnoc-ls0.1-ow0.0-1.0-1.0-cams-0.2-octis1-amp@rs269ra-r3@$DOMAIN@scale=0.5,1.0,1.5,2.0
RW_DIR=$MODEL_NAME@$DOMAIN@beta=10@exp_times=8@rw
# run_inference
# $PY scripts/evaluate.py --experiment_name $RW_DIR --dataset voc12 --domain val --data_dir $DATA_DIR --min_th 0.05 --max_th 0.81 --crf_t 1 --crf_gt_prob 0.9

MODEL_NAME=rw/voc12-an@ccamh@rs269pnoc-ls0.1@fg0.3-bg0.1-crf10-gt0.7
CAM_DIR=pnoc/voc12-rs269-pnoc-ls0.1-ow0.0-1.0-1.0-cams-0.2-octis1-amp@rs269ra-r3@$DOMAIN@scale=0.5,1.0,1.5,2.0
RW_DIR=$MODEL_NAME@$DOMAIN@beta=10@exp_times=8@rw
# run_inference
# $PY scripts/evaluate.py --experiment_name $RW_DIR --dataset voc12 --domain val --data_dir $DATA_DIR --min_th 0.05 --max_th 0.81 --crf_t 1 --crf_gt_prob 0.9


MODEL_NAME=rw/affnet@rs269-poc@pn-fgh@crf-10-gt-0.9@aff_fg=0.40_bg=0.10
CAM_DIR=poc/ResNeSt269@PuzzleOc@$DOMAIN@scale=0.5,1.0,1.5,2.0
RW_DIR=$MODEL_NAME@$DOMAIN@beta=10@exp_times=8@rw
# run_inference
# $PY scripts/evaluate.py --experiment_name $RW_DIR --dataset voc12 --domain val --data_dir $DATA_DIR --min_th 0.05 --max_th 0.81 --crf_t 1 --crf_gt_prob 0.9


## =========================================
## MS COCO 14 Dataset
## =========================================

DATASET=coco14
IMAGE_SIZE=640
DATA_DIR=/home/ldavid/workspace/datasets/coco14/

MODEL_NAME=rw/coco14-an-640@pnoc-lr0.05-ccamh-ls@rs269ra

DOMAIN=train2014
CAM_DIR=pnoc/coco14-rs269-pnoc-b16-a2-lr0.05-ls0-ow0.0-1.0-1.0-c0.2-is1@rs269ra-r1@train@scale=0.5,1.0,1.5,2.0
RW_DIR=$MODEL_NAME@train@beta=10@exp_times=8@rw
# run_inference

TAG=rw/coco14-an-640@pnoc-lr0.05-ccamh-ls@rs269ra@train@beta=10@exp_times=8@rw
THRESHOLD=0.3
CRF_T=1
CRF_GT=0.9
run_crf

DOMAIN=val2014
CAM_DIR=pnoc/coco14-rs269-pnoc-b16-a2-lr0.05-ls0-ow0.0-1.0-1.0-c0.2-is1@rs269ra-r1@val@scale=0.5,1.0,1.5,2.0
RW_DIR=$MODEL_NAME@val@beta=10@exp_times=8@rw
# run_inference

