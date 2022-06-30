#!/bin/bash
#SBATCH --nodes=1
#SBATCH -p nvidia_long
#SBATCH -J puzzlecam-rw
#SBATCH -o /scratch/lerdl/lucas.david/logs/puzzlecam/%j.randomwalk.out
#SBATCH --time=12:00:00

echo "[puzzlecam/2-randomwalk] started running at $(date +'%Y-%m-%d %H:%M:%S')."

nodeset -e $SLURM_JOB_NODELIST

module load gcc/7.4 python/3.7.2 cudnn/8.2_cuda-11.1

cd $SCRATCH/PuzzleCAM
source $SCRATCH/envs/torch/bin/activate
# pip -qq install -r requirements.txt
# torch==1.3.0
# torchvision==0.4.1

CUDA_VISIBLE_DEVICES=0 python3.7 inference_classification.py \
    --architecture resnest101 \
    --tag ResNeSt101@Puzzle@optimal \
    --domain train_aug \
    --data_dir $SCRATCH/datasets/voc/VOCdevkit/VOC2012/

# python3 make_affinity_labels.py \
#     --experiment_name ResNeSt101@Puzzle@optimal@train@scale=0.5,1.0,1.5,2.0 \
#     --domain train_aug  \
#     --fg_threshold 0.40 \
#     --bg_threshold 0.10 \
#     --data_dir $SCRATCH/datasets/voc/VOCdevkit/VOC2012/
