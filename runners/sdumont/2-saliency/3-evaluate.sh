#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH -p nvidia_long
#SBATCH -J ev-ccam
#SBATCH -o /scratch/lerdl/lucas.david/logs/ccam/ev-%j.out
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

TAG=saliency/coco14-ccamh-rs269@rs269pnoc@rs269@b64-fg0.3-lr0.0005-b64@train@scale=0.5,1.0,1.5,2.0
DOMAIN=val2014
CRF_T=10

$PY $SOURCE                   \
  --experiment_name $TAG      \
  --min_th          0.05      \
  --max_th          0.51      \
  --crf_t           $CRF_T    \
  --dataset         $DATASET  \
  --domain          $DOMAIN   \
  --data_dir        $DATA_DIR \
  --num_workers     $WORKERS

