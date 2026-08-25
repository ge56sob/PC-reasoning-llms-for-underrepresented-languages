#!/bin/bash
#SBATCH --partition lrz-hgx-h100-94x4
#SBATCH --gres gpu:1
#SBATCH --time=48:00:00
#SBATCH --output %j.out

python -m pip install --user datasets
python -m pip install --user transformers
python -m pip install --user accelerate sentencepiece

export HF_HOME=/dss/dssfs05/lwp-dss-0003/pn39je/pn39je-dss-0004/go35wit2/huggingface
export HF_HUB_CACHE=$HF_HOME/models
export HF_DATASETS_CACHE=$HF_HOME/datasets

python Cuong_LargeModel.py

git add -A && \
git commit -m "automated push" && \
git push
