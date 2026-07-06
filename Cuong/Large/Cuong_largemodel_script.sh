#!/bin/bash
#SBATCH --partition lrz-hgx-h100-94x4
#SBATCH --gres gpu:1
#SBATCH --time=12:00:00
#SBATCH --output %j.out

python -m pip install datasets
python -m pip install llama-cpp-python


python Cuong_LargeModel.py

git add -A && \
git commit -m "automated push" && \
git push
