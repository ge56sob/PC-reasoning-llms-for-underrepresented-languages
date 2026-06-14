#!/bin/bash
#SBATCH --partition lrz-hgx-h100-94x4
#SBATCH --gres gpu:1
#SBATCH --time=12:00:00
#SBATCH --output %j.out

python cuong_small_model_3.py
