#!/bin/bash
#SBATCH --nodes=1
#SBATCH -p nvidia_long
#SBATCH -J puzzlecam-rw
#SBATCH -o /scratch/lerdl/lucas.david/logs/puzzlecam/%j.randomwalk.out
#SBATCH --time=12:00:00

echo "[segm/voc12.rn50.sh] started running at $(date +'%Y-%m-%d %H:%M:%S')."

nodeset -e $SLURM_JOB_NODELIST

module load gcc/7.4 python/3.9.1 cudnn/8.2_cuda-11.1

cd $SCRATCH/PuzzleCAM
source $SCRATCH/envs/torch/bin/activate

pip install -r requirements.txt

module load $MODULES

python3 make_affinity_labels.py \
    --experiment_name ResNeSt101@Puzzle@optimal@train@scale=0.5,1.0,1.5,2.0 \
    --domain train_aug  \
    --fg_threshold 0.40 \
    --bg_threshold 0.10 \
    --data_dir $SCRATCH/datasets/voc/VOCdevkit
