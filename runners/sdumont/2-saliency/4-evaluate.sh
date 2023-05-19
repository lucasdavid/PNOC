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
module load gcc/7.4 python/3.9.1 cudnn/8.2_cuda-11.1


export PYTHONPATH=$(pwd)

PY=python3.9
SOURCE=scripts/ccam/evaluate.py
WORKERS=48

DATASET=coco14
DATA_DIR=$SCRATCH/datasets/coco14/
DOMAIN=train2014

# TAG=saliency/coco14-pn@ccamh-rs269-fg0.2@rs269pnoc-ls0.1
TAG=saliency/coco14-pn@ccamh-rs269-fg0.25@rs269pnoc-lr0.05-ls0.1
CRF_T=10
CRF_GT_PROB=0.7

WANDB_TAGS="$DATASET,domain:$DOMAIN,ccamh" \
$PY $SOURCE                   \
  --experiment_name $TAG      \
  --dataset         $DATASET  \
  --domain          $DOMAIN   \
  --min_th          0.2       \
  --max_th          1.0       \
  --mode            png       \
  --eval_mode       segmentation \
  --crf_t           $CRF_T       \
  --crf_gt_prob     $CRF_GT_PROB \
  --data_dir        $DATA_DIR    \
  --num_workers     $WORKERS

