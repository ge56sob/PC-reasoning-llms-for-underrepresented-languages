#!/bin/bash
#SBATCH --partition lrz-hgx-h100-94x4
#SBATCH --gres gpu:1
export TENSOR_PARALLEL_SIZE=2
#SBATCH --time 2-00:00:00
#SBATCH --output %j.out

export HF_TOKEN=
export HUGGINGFACE_HUB_TOKEN=$HF_TOKEN

export BIGDIR=/dss/dssfs05/lwp-dss-0003/pn39je/pn39je-dss-0004/go35wit2

export HF_HOME=$BIGDIR/huggingface
export HF_HUB_CACHE=$BIGDIR/huggingface/hub
export TRANSFORMERS_CACHE=$BIGDIR/huggingface/transformers
export HF_DATASETS_CACHE=$BIGDIR/huggingface/datasets
export XDG_CACHE_HOME=$BIGDIR/.cache
export TMPDIR=$BIGDIR/tmp

mkdir -p $HF_HOME $HF_HUB_CACHE $TRANSFORMERS_CACHE $HF_DATASETS_CACHE $XDG_CACHE_HOME $TMPDIR

cd $BIGDIR 

export HF_HUB_DISABLE_XET=1

python3 -u Cuong_LargeModel.py

git add -A && \
git commit -m "automated push" && \
git push
