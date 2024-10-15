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

  PY=python
  PIP=pip
  DEVICE=cuda
  DEVICES=0,1
  GPUS=all
  WORKERS_TRAIN=8
  WORKERS_INFER=8
  DATASETS_DIR=$SCRATCH/datasets
elif [[ $ENV == cenapad ]]; then
  
  echo "Loading modules cudnn/8.2.0.53-11.3-gcc-9.3.0 python/3.8.11-gcc-9.4.0"
  module load cudnn/8.2.0.53-11.3-gcc-9.3.0; module load python/3.8.11-gcc-9.4.0

  DATA_DIR=$SCRATCH/datasets
  BUILD_DIR=$SCRATCH/build

  ### CENAPAD Lovelace
  PY=python
  PIP=pip
  DEVICE=cuda
  DEVICES=0,1
  WORKERS_TRAIN=4
  WORKERS_INFER=32
  DATASETS_DIR=$SCRATCH/datasets

  # export OMP_NUM_THREADS=4
elif [[ $ENV == cenapad-umagpu ]]; then

  echo "Loading modules cudnn/8.2.0.53-11.3-gcc-9.3.0 python/3.8.11-gcc-9.4.0"
  module load cudnn/8.2.0.53-11.3-gcc-9.3.0; module load python/3.8.11-gcc-9.4.0

  DATA_DIR=$SCRATCH/datasets
  BUILD_DIR=$SCRATCH/build

  ### CENAPAD Lovelace
  PY=python
  PIP=pip
  DEVICE=cuda
  DEVICES=0
  WORKERS_TRAIN=2
  WORKERS_INFER=16
  DATASETS_DIR=$SCRATCH/datasets

  # export OMP_NUM_THREADS=4
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

