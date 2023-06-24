#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH -p nvidia_long
#SBATCH -J ev
#SBATCH -o /scratch/lerdl/lucas.david/logs/ccam/ev-%j.out
#SBATCH --time=8:00:00


echo "[sdumont/sequana/classification/train-puzzle] started running at $(date +'%Y-%m-%d %H:%M:%S')."

nodeset -e $SLURM_JOB_NODELIST

cd $SCRATCH/PuzzleCAM

# module load sequana/current
# module load gcc/7.4_sequana python/3.9.1_sequana cudnn/8.2_cuda-11.1_sequana
# module load gcc/7.4 python/3.9.1 cudnn/8.2_cuda-11.1
module load gcc/7.4 python/3.8.2 cudnn/8.2_cuda-11.1


export PYTHONPATH=$(pwd)

PY=python3.8
SOURCE=scripts/ccam/evaluate.py
WORKERS=48

DATASET=voc12
DATA_DIR=$SCRATCH/datasets/VOCdevkit/VOC2012/
DOMAIN=train

run_evaluation() {
  WANDB_TAGS="$DATASET,domain:$DOMAIN,ccamh,crf:$CRF_T-$CRF_GT_PROB" \
  $PY $SOURCE                   \
    --experiment_name $TAG      \
    --dataset         $DATASET  \
    --domain          $DOMAIN   \
    --min_th          0.2       \
    --max_th          0.8       \
    --mode            npy       \
    --eval_mode       saliency  \
    --crf_t           $CRF_T       \
    --crf_gt_prob     $CRF_GT_PROB \
    --data_dir        $DATA_DIR    \
    --num_workers     $WORKERS
}

CRF_T=10
CRF_GT_PROB=0.7

TAG=saliency/voc12-ccamh-rs269@ra-oc-p-poc-pnoc-avg@b64-fg0.3-lr0.001-b64@train@scale=0.5,1.0,1.5,2.0
run_evaluation
TAG=saliency/voc12-ccamh-rs269@ra-oc-p-poc-pnoc-learned-a0.25@b64-fg0.3-lr0.001-b64@train@scale=0.5,1.0,1.5,2.0
run_evaluation
TAG=saliency/voc12-ccamh-rs269@ra-oc-p-poc-pnoc-weighted-a0.25@b64-fg0.3-lr0.001-b64@train@scale=0.5,1.0,1.5,2.0
run_evaluation


## ====================
## Dataset MS COCO 2014
## ====================

# DATASET=coco14
# DATA_DIR=$SCRATCH/datasets/coco14/
# DOMAIN=train2014

# TAG=saliency/coco14-pn@ccamh-rs269-fg0.2@rs269pnoc-ls0.1
# run_evaluation

# TAG=saliency/coco14-pn@ccamh-rs269-fg0.25@rs269pnoc-lr0.05-ls0.1
# run_evaluation
