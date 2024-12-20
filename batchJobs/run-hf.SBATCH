#!/bin/bash

## change the last two digits to your team id
#SBATCH --account=csci_ga_2271_001-2024fa

# change the partition number to use different number of GPUs
#SBATCH --partition=n1s8-v100-1
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8

##SBATCH --partition=n1s16-v100-2
##SBATCH --gres=gpu:2
##SBATCH --cpus-per-task=16

##SBATCH --partition=n1c24m128-v100-4
##SBATCH --gres=gpu:4
##SBATCH --cpus-per-task=24

#SBATCH --time=2:00:00
#SBATCH --output=cvproject_%j.out
#SBATCH --error=cvproject_%j.err
##SBATCH --exclusive
#SBATCH --requeue

singularity exec --bind /scratch/sc9960 --nv \
            --overlay /scratch/sc9960/cv_project/g3-variant/singularity-image/g3-gg.ext3:ro \
            /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "source /ext3/env.sh; 
            cd /scratch/sc9960/cv_project/g3-variant/g3-gg; 
            python llm_predict_hf.py"
