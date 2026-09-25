#!/usr/bin/env bash
set -e

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

python3 -u train.py \
  --data "$HOME/Model-Az/AC-model/data/training_shards" \
        "$HOME/Model-Az/AC-model/data/combined_train" \
  --tokenizer "$HOME/Model-Az/AE-model/data/bpe_tokenizer.json" \
  --seq-len 256 \
  --dim 256 \
  --layers 12 \
  --heads 8 \
  --kv-heads 2 \
  --ffn 768 \
  --batch 2 \
  --grad-accum 8 \
  --steps 10000 \
  --lr 3e-4 \
  --warmup 500 \
  --workers 1 \
  --save-every 500
