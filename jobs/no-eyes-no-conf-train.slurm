#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=00:30:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --mem=12000  
#SBATCH --job-name="stgcn-no-eyes-no-conf-train"
#SBATCH --output=stgcn.out

module purge
module load GCC/7.3.0-2.30  CUDA/9.2.88  OpenMPI/3.1.1
module load Python/3.6.6

source stgcn/bin/activate
cd ../../lustre1/work/sindreau/mmskeleton

echo "start"
python mmskl.py configs/custom/no-eyes-no-conf/train.yaml
echo "slutt"