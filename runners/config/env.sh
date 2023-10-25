# Configurations

## Environment

if [[ $ENV == sdumont ]]; then
  ### Sdumont
  PY=python3.8
  PIP=pip3.8
  DEVICE=cuda
  DEVICES=0,1,2,3
  WORKERS_TRAIN=8
  WORKERS_INFER=48
  DATASETS_DIR=$SCRATCH/datasets

  # export OMP_NUM_THREADS=4
  nodeset -e $SLURM_JOB_NODELIST
  module load sequana/current
  module load gcc/7.4_sequana python/3.8.2_sequana cudnn/8.2_cuda-11.1_sequana
elif [[ $ENV == sdumont_nvidia ]]; then
  nodeset -e $SLURM_JOB_NODELIST

  echo "Loading modules deepl/deeplearn-py3.7"
  module load deepl/deeplearn-py3.7

  DATA_DIR=$SCRATCH/datasets
  BUILD_DIR=$SCRATCH/single-stage/build

  PIP=pip
  PY=python
  DEVICE=cuda
  DEVICES=0,1
  GPUS=all
  WORKERS_TRAIN=8
  WORKERS_INFER=8
  DATASETS_DIR=$SCRATCH/datasets
else
  ### Local
  PY=python
  PIP=pip
  DEVICE=cuda
  DEVICES=0
  WORKERS_TRAIN=8
  WORKERS_INFER=8
  DATASETS_DIR=$HOME/workspace/datasets
fi
