#!/bin/sh
#SBATCH --job-name=Gemma2  # create a short name for your job
#SBATCH --partition=gpu-long  # config basic job type: https://dochub.comp.nus.edu.sg/cf/guides/compute-cluster/slurm-info
#SBATCH --time=5-00:0:00      # set your running time to 5d
#SBATCH --gres=gpu:h100-96:1  # config gpu: https://dochub.comp.nus.edu.sg/cf/guides/compute-cluster/gpu
# ~/miniconda3/ is the python environment created in step 1
~/miniconda3/bin/wandb login 4bdc5d5b0ba67a677079a186c2fd8c304d9453ff # if you need wandb
~/miniconda3/bin/python /home/i/i0001574/workspace/alpha/TokenizationFairnessTesting/batch_all.sh  # this is the script you’ve tested in the adhoc session