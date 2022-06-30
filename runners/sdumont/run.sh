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
your_dir=$SCRATCH/datasets/voc/VOCdevkit/VOC2012/

## 1. Train an image classifier for generating CAMs
CUDA_VISIBLE_DEVICES=0,1,2,3 python3.7 train_classification_with_puzzle.py --architecture resnest101 --re_loss_option masking --re_loss L1_Loss --alpha_schedule 0.50 --alpha 4.00 --tag ResNeSt101@Puzzle@optimal --data_dir $your_dir

## 2. Apply Random Walk (RW) to refine the generated CAMs
CUDA_VISIBLE_DEVICES=0 python3.7 inference_classification.py --architecture resnest101 --tag ResNeSt101@Puzzle@optimal --domain train_aug --data_dir $your_dir
CUDA_VISIBLE_DEVICES=0 python3.7 make_affinity_labels.py --experiment_name ResNeSt101@Puzzle@optimal@train@scale=0.5,1.0,1.5,2.0 --domain train_aug --fg_threshold 0.40 --bg_threshold 0.10 --data_dir $your_dir

CUDA_VISIBLE_DEVICES=0 python3.7 train_affinitynet.py --architecture resnest101 --tag AffinityNet@ResNeSt-101@Puzzle --label_name ResNeSt101@Puzzle@optimal@train@scale=0.5,1.0,1.5,2.0@aff_fg=0.40_bg=0.10 --data_dir $your_dir

## 3. Train the segmentation model using the pseudo-labels
CUDA_VISIBLE_DEVICES=1 python3.7 inference_rw.py --architecture resnest101 --model_name AffinityNet@ResNeSt-101@Puzzle --cam_dir ResNeSt101@Puzzle@optimal@train@scale=0.5,1.0,1.5,2.0 --domain train_aug --data_dir $your_dir

CUDA_VISIBLE_DEVICES=1 python3.7 make_pseudo_labels.py --experiment_name AffinityNet@ResNeSt-101@Puzzle@train@beta=10@exp_times=8@rw --domain train_aug --threshold 0.35 --crf_iteration 1 --data_dir $your_dir

# 3.2. Train segmentation model.
CUDA_VISIBLE_DEVICES=0,1,2,3 python3.7 train_segmentation.py --backbone resnest101 --mode fix --use_gn True --tag DeepLabv3+@ResNeSt-101@Fix@GN --label_name AffinityNet@ResNeSt-101@Puzzle@train@beta=10@exp_times=8@rw@crf=1 --data_dir $your_dir

## 4. Evaluate the models
CUDA_VISIBLE_DEVICES=0 python3.7 inference_segmentation.py --backbone resnest101 --mode fix --use_gn True --tag DeepLabv3+@ResNeSt-101@Fix@GN --scale 0.5,1.0,1.5,2.0 --iteration 10

python3.7 evaluate.py --experiment_name DeepLabv3+@ResNeSt-101@Fix@GN@val@scale=0.5,1.0,1.5,2.0@iteration=10 --domain val --data_dir $your_dir/SegmentationClass
