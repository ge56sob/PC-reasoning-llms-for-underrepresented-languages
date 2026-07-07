#!/bin/bash
#SBATCH --partition lrz-hgx-h100-94x4
#SBATCH --gres gpu:1
#SBATCH --time=12:00:00
#SBATCH --output %j.out

python -m pip install --user datasets
python -m pip install --user transformers
python -m pip install --user accelerate sentencepiece

python Cuong_Backtranslation_Small.py

git add . && \
git commit -m "automated push" && \
git push
