#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH -p nvidia_long
#SBATCH -J ev
#SBATCH -o /scratch/lerdl/lucas.david/logs/evaluate-%j.out
#SBATCH --time=48:00:00


echo "[sdumont/sequana/classification/train-puzzle] started running at $(date +'%Y-%m-%d %H:%M:%S')."

nodeset -e $SLURM_JOB_NODELIST

cd $SCRATCH/PuzzleCAM

# module load sequana/current
# module load gcc/7.4_sequana python/3.9.1_sequana cudnn/8.2_cuda-11.1_sequana
module load gcc/7.4 python/3.9.1 cudnn/8.2_cuda-11.1


export PYTHONPATH=$(pwd)

PY=python3.9
SOURCE=scripts/evaluate.py
WORKERS=48

DATASET=coco14
DATA_DIR=$SCRATCH/datasets/coco14/
DOMAIN=train2014
# DOMAIN=val2014
VERBOSE=1

MIN_TH=0.15
MAX_TH=0.51


run_inference() {
  $PY $SOURCE                   \
    --experiment_name $TAG      \
    --dataset         $DATASET  \
    --domain          $DOMAIN   \
    --data_dir        $DATA_DIR \
    --min_th          $MIN_TH   \
    --max_th          $MAX_TH   \
    --crf_t           $CRF_T       \
    --crf_gt_prob     $CRF_GT_PROB \
    --verbose         $VERBOSE     \
    --num_workers     $WORKERS
}

run_inference_sal() {
  $PY $SOURCE                   \
    --experiment_name $TAG      \
    --dataset         $DATASET  \
    --domain          $DOMAIN   \
    --data_dir        $DATA_DIR \
    --sal_dir         $SAL_DIR  \
    --min_th          $MIN_TH   \
    --max_th          $MAX_TH   \
    --crf_t           $CRF_T       \
    --crf_gt_prob     $CRF_GT_PROB \
    --verbose         $VERBOSE     \
    --num_workers     $WORKERS
}

# TAG=pnoc/coco14-rs269-pnoc-b16-a2-ls0.1-ow0.0-1.0-1.0-c0.2-is1@rs269ra-r3@train@scale=0.5,1.0,1.5,2.0
# run_inference

# SAL_DIR="./experiments/predictions/saliency/coco14-pn@ccamh-rs269-fg0.2@rs269pnoc-ls0.1"
# CRF_T=10
# CRF_GT_PROB=0.7
# run_inference_sal

# TAG=rw/coco14-an-640@pnoc-ls0.1-ccamh-ls0.1@rs269ra@train@beta=10@exp_times=8@rw
# MIN_TH=0.25
# MAX_TH=0.46
# CRF_T=1
# CRF_GT_PROB=0.9
# run_inference

# TAG=pnoc/coco14-rs269-pnoc-b16-a2-lr0.05-ls0-ow0.0-1.0-1.0-c0.2-is1@rs269ra-r1@val@scale=0.5,1.0,1.5,2.0
TAG=pnoc/coco14-rs269-pnoc-b16-a2-lr0.05-ls0.1-ow0.0-1.0-1.0-c0.2-is1@rs269ra-r1@val@scale=0.5,1.0,1.5,2.0
DOMAIN=val2014
CRF_GT_PROB=0.7
CRF_T=0
run_inference

# CRF_T=10
# CRF_GT_PROB=0.7
# run_inference

CRF_T=10
CRF_GT_PROB=0.7
# run_inference

TAG=rw/coco14-an-640@pnoc-lr0.05-ccamh-ls@rs269ra@val@beta=10@exp_times=8@rw
CRF_T=1
CRF_GT_PROB=0.9
run_inference
