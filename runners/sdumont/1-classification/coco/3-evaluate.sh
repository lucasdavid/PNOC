#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH -p nvidia_long
#SBATCH -J ev
#SBATCH -o /scratch/lerdl/lucas.david/logs/evaluate-%j.out
#SBATCH --time=02:00:00


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

TAG=pnoc/coco14-rs269-pnoc-b16-a2-ls0.1-ow0.0-1.0-1.0-c0.2-is1@rs269ra-r3@train@scale=0.5,1.0,1.5,2.0
SAL_DIR="./experiments/predictions/saliency/coco14-pn@ccamh-rs269-fg0.2@rs269pnoc-ls0.1"

CRF_T=10
CRF_GT_PROB=0.7

$PY $SOURCE                   \
  --experiment_name $TAG      \
  --dataset         $DATASET  \
  --domain          $DOMAIN   \
  --data_dir        $DATA_DIR \
  --sal_dir         $SAL_DIR  \
  --crf_t           $CRF_T       \
  --crf_gt_prob     $CRF_GT_PROB \
  --num_workers     $WORKERS
