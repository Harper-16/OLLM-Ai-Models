# Ultimate CPU LLM for an 8 GB laptop

This is a compact GPT-style causal Transformer designed around a CPU-only laptop with about 8 GB RAM.

Architecture:
- decoder-only causal Transformer
- RoPE
- RMSNorm
- Grouped-Query Attention (GQA)
- PyTorch scaled-dot-product attention
- SwiGLU feed-forward blocks
- tied token/output embeddings
- residual connections
- AdamW + warmup + cosine decay
- gradient accumulation
- streaming dataset, so the corpus is not loaded into RAM
- resumable checkpoints
- optional SFT stage for instruction following

Default model is roughly 28M parameters when using a 32k tokenizer. It is intentionally much smaller than modern ChatGPT-scale models because an 8 GB CPU laptop cannot practically pretrain a hundreds-of-millions/billions parameter model.

## Install

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

## Use your existing AC-model data

Copy this project into `~/Model-Az/AC-model/ultimate_cpu` or run it there.

Expected paths:
- data/training_shards/*.txt
- data/combined_train/*.txt or a text file
- data/sharded_model/tokenizer.json

The tokenizer is loaded from your existing tokenizer.json when the `tokenizers` package can read it. Otherwise the code falls back to a byte tokenizer.

## Train

bash run_cpu.sh

Resume:

python3 train.py --resume runs/ultimate/latest.pt [other arguments]

## Chat

python3 chat.py --checkpoint runs/ultimate/final.pt

## Important

Pretraining creates a base language model. It does NOT automatically make a ChatGPT-like assistant. For instruction following, do SFT after pretraining with JSONL instruction/response data.

The model can learn language patterns, but its quality depends heavily on token count, data quality, tokenizer, model size, and training compute.
# OLLM-Ai-Models
