#!/bin/bash

#SBATCH --job-name=deep_gan
#SBATCH --output=gan_output.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=4
#SBATCH --partition=gpu
#SBATCH --time=2-

module load miniconda
module load cuDNN/7.6.2.24-CUDA-10.0.130
conda activate torch11

python train.py --epoch=50 --experiment_name=gan
python train.py --epoch=50 --adversarial_loss_mode=lsgan --experiment_name=lsgan
python train.py --epoch=50 --adversarial_loss_mode=wgan --gradient_penalty_mode=1-gp --gradient_penalty_sample_mode=line --n_d=5 --experiment_name=wgan


