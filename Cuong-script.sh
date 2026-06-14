#!/bin/bash
#SBATCH --partition lrz-hgx-h100-94x4
#SBATCH --gres gpu:1
#SBATCH --time 2-00:00:00
#SBATCH --dependency afterok:<prev_jobid>
#SBATCH --output %j.out

python cuong_small_model_3.py
