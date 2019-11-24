#!/bin/bash

# Slurm submission script
# GPU job
# CEDAR: https://docs.computecanada.ca/wiki/Cedar
# Doc slurm: https://docs.computecanada.ca/wiki/Running_jobs
# Doc slurm: https://docs.computecanada.ca/wiki/Using_GPUs_with_Slurm
# bach-part-a-2018: 16GB
# fgnet: 24GB


#SBATCH --account=def-egranger
#SBATCH --output=./outputjobs/o-c%J.o
#SBATCH --error=./outputjobs/o-c%J.e
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
# SBATCH --mem-per-cpu=1024M
#SBATCH --mem=24000M
#SBATCH --mail-user=soufiane.belharbi.1@etsmtl.net
#SBATCH --mail-type=ALL
# SBATCH --time=03:00:00

# workon pytorch.1.0.1
# source $HOME/Venvs/pytorch.1.0.1/bin/activate
# module load cuda/10.0.130


#SBATCH --time=0-2:30

source $HOME/Venvs/pytorch.1.2.0/bin/activate 
module load cuda/10.0.130

python main.py --cudaid 0 --yaml bach-part-a-2018.yaml --bsize 8 --lr 0.001 --wdecay 1e-05 --momentum 0.9 --epoch 1000 --stepsize 100 --modelname resnet18 --alpha 0.6 --kmax 0.1 --kmin 0.1 --dout 0.0 --modalities 5 --pretrained True  --dataset bach-part-a-2018 --split 0 --fold 0  --loss LossCE  
