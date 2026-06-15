#!/bin/bash
#SBATCH --partition lrz-hgx-h100-94x4
#SBATCH --gres gpu:1
#SBATCH --time=12:00:00
#SBATCH --output %j.out

python -m pip install --user huggingface_hub
python -m pip install --user datasets
python -m pip install --user transformers==4.52.4
python -m pip install --user accelerate sentencepiece

python cuong_small_model_3.py

git add -A && \
git commit -m "automated push" && \
git push
