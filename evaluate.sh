#!/bin/bash

#SBATCH -p a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err

module load miniconda3
source activate
conda activate common

export XDG_RUNTIME_DIR=/dssg/home/acct-stu/stu513/ai3611/av_scene_classify/plt_img

# evaluation
python evaluate.py --experiment_path experiments/baseline