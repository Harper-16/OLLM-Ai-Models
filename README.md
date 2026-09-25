# AE-model: Ultimate CPU LLM

A compact, GPT-style causal language model designed for **CPU-only training and inference on laptops with around 8 GB of RAM**.

The goal is simple: build a real Transformer from the ground up that can be trained, resumed, fine-tuned, and used locally without requiring a large GPU.

> **Project philosophy:** small enough to experiment with, but complete enough to demonstrate the major components of a modern decoder-only LLM.

## Architecture

AE-model uses a decoder-only Transformer with:

* **Causal self-attention** for autoregressive next-token prediction
* **RoPE** (Rotary Positional Embeddings)
* **RMSNorm** for stable normalization
* **GQA** (Grouped-Query Attention) to reduce KV-cache and attention memory
* **PyTorch scaled-dot-product attention**
* **SwiGLU** feed-forward networks
* **Tied input/output embeddings** to reduce parameter count
* **Residual connections** throughout the Transformer
* **AdamW** optimizer
* **Linear warmup + cosine learning-rate decay**
* **Gradient accumulation** for effectively larger batch sizes
* **Streaming dataset loading** so the complete corpus does not need to fit in RAM
* **Resumable checkpoints** for long CPU training runs
* **Optional supervised fine-tuning (SFT)** for instruction following

### Default model

The default configuration is approximately **28 million parameters** with a 32k-token tokenizer.

This is deliberately small.

A model of this size will not have the capabilities of large commercial LLMs, but it is small enough to make the complete training pipeline practical on consumer hardware and useful for experimenting with LLM architecture and training.

---

## What AE-model actually learns

During pretraining, the model learns to predict the next token:

```text
The cat sat on the → ?
```

The model is therefore learning **language patterns**, not automatically becoming a conversational assistant.

Pretraining can teach it:

* vocabulary and token relationships
* syntax
* common language patterns
* information present in the training corpus
* text continuation
* basic stylistic patterns

However, instruction following is a separate stage.

For example:

```text
User: Explain photosynthesis.

Assistant:
...
```

requires instruction-style training data and supervised fine-tuning.

---

# Requirements

Recommended minimum:

* Linux
* Python 3.10+
* ~8 GB RAM
* CPU with multiple cores
* Several GB of free disk space

A GPU is **not required**.

Training will be much slower on CPU than on a modern GPU, so this project is primarily intended for experimentation, learning, and small-scale model development.

---

# Installation

Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Dataset

AE-model can use the existing **AC-model** training data.

Expected data structure:

```text
data/
├── training_shards/
│   ├── shard_000.txt
│   ├── shard_001.txt
│   └── ...
│
├── combined_train/
│   └── *.txt
│
└── sharded_model/
    └── tokenizer.json
```

The exact dataset layout depends on which training pipeline you are using.

The training loader is designed to **stream text instead of loading the entire corpus into RAM**.

This is important for an 8 GB machine.

---

# Tokenizer

AE-model can use an existing tokenizer from:

```text
data/sharded_model/tokenizer.json
```

When the `tokenizers` package can load the tokenizer, AE-model uses it directly.

If the tokenizer cannot be loaded, the project can fall back to a byte-level tokenizer.

For best results, use the same tokenizer consistently throughout training and inference.

---

# Training

The simplest training command is:

```bash
bash run_cpu.sh
```

The script starts CPU training using the project's default configuration.

Training produces checkpoints under:

```text
runs/ultimate/
```

For example:

```text
runs/ultimate/
├── config.json
├── latest.pt
└── final.pt
```

---

# Resuming Training

Long CPU training runs can take a significant amount of time.

AE-model therefore supports resumable checkpoints.

Resume from the latest checkpoint:

```bash
python3 train.py --resume runs/ultimate/latest.pt
```

You can also provide the other training arguments required by your configuration.

A checkpoint can contain the model state, optimizer state, scheduler state, and training progress, allowing training to continue rather than starting from zero.

---

# Chat / Inference

After training, launch the model with:

```bash
python3 chat.py --checkpoint runs/ultimate/final.pt
```

You can also use another checkpoint:

```bash
python3 chat.py --checkpoint runs/ultimate/latest.pt
```

The tokenizer used for inference should match the tokenizer used during training.

---

# Supervised Fine-Tuning

Pretraining creates a **base language model**.

It does not automatically create a polished ChatGPT-style assistant.

For instruction following, use the optional SFT stage.

A typical SFT dataset can contain JSONL records such as:

```json
{"instruction":"Explain gravity in simple terms.","response":"Gravity is the force that attracts objects toward one another."}
```

SFT teaches the pretrained model how to respond to instructions rather than simply continuing arbitrary text.

The general pipeline is:

```text
Raw text
   ↓
Tokenizer
   ↓
Pretraining
   ↓
Base language model
   ↓
Instruction / response dataset
   ↓
SFT
   ↓
Instruction-following model
```

---

# Training Pipeline

The complete AE-model workflow is:

```text
              ┌─────────────────┐
              │   Text Dataset  │
              └────────┬────────┘
                       ↓
              ┌─────────────────┐
              │    Tokenizer    │
              └────────┬────────┘
                       ↓
              ┌─────────────────┐
              │   Pretraining   │
              │  Causal LM      │
              └────────┬────────┘
                       ↓
              ┌─────────────────┐
              │  Base Model     │
              └────────┬────────┘
                       ↓
              ┌─────────────────┐
              │       SFT       │
              └────────┬────────┘
                       ↓
              ┌─────────────────┐
              │ Instruction     │
              │ Following Model │
              └─────────────────┘
```

---

# Memory Efficiency

The project is specifically designed around the limitations of an 8 GB CPU laptop.

Several choices help keep memory usage manageable:

### Streaming data

The complete corpus is not loaded into RAM.

Instead, training data is read progressively from disk.

### GQA

Grouped-Query Attention reduces the number of key/value heads compared with standard multi-head attention.

### Gradient accumulation

Multiple smaller micro-batches can be accumulated to simulate a larger effective batch size without requiring the entire batch to fit in memory simultaneously.

### Small model

The default configuration intentionally stays around the tens-of-millions parameter range.

This makes experimentation possible on hardware where a much larger model would be impractical.

---

# Checkpoints

Checkpoints are saved during training so long-running CPU experiments can be resumed.

Typical files:

```text
runs/ultimate/
├── config.json
├── latest.pt
└── final.pt
```

`latest.pt` is intended for continuing training.

`final.pt` represents the final saved training state.

Because PyTorch checkpoint files can become large, avoid committing oversized checkpoint files directly to normal Git history when they exceed your Git hosting provider's file limits.

---

# Model Quality

Model quality depends heavily on:

1. **Number of training tokens**
2. **Dataset quality**
3. **Dataset diversity**
4. **Tokenizer quality**
5. **Model size**
6. **Context length**
7. **Training duration**
8. **Learning-rate schedule**
9. **Data cleanliness**
10. **Instruction-tuning quality**

A 28M parameter model should be viewed as a **small experimental language model**, not as a replacement for large-scale LLMs.

More parameters alone do not guarantee better results. Training data and optimization matter enormously.

---

# Why build a small LLM?

Large models hide a lot of the machinery behind APIs and enormous compute budgets.

AE-model is intended to make that machinery understandable and editable.

You can inspect and experiment with:

```text
Tokenizer
   ↓
Token IDs
   ↓
Embeddings
   ↓
RoPE
   ↓
GQA Attention
   ↓
SwiGLU
   ↓
Residual connections
   ↓
RMSNorm
   ↓
Next-token probabilities
```

The entire pipeline is small enough to study and modify.

---

# Project Status

AE-model is an experimental CPU-focused LLM project.

Current capabilities include:

* [x] Decoder-only Transformer
* [x] Causal language modeling
* [x] RoPE
* [x] RMSNorm
* [x] GQA
* [x] SwiGLU
* [x] Tied embeddings
* [x] Streaming dataset
* [x] AdamW
* [x] Learning-rate warmup
* [x] Cosine decay
* [x] Gradient accumulation
* [x] Checkpointing
* [x] Resume training
* [x] Local chat/inference
* [x] Optional SFT pipeline

---

# Example Commands

### Install

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Train

```bash
bash run_cpu.sh
```

### Resume

```bash
python3 train.py --resume runs/ultimate/latest.pt
```

### Chat

```bash
python3 chat.py --checkpoint runs/ultimate/final.pt
```

---

# Hardware Philosophy

AE-model is built around an intentionally constrained target:

> **What useful LLM experiments can we run on an ordinary 8 GB CPU laptop?**

That constraint shapes the architecture, dataset loader, checkpoint system, and training configuration.

It will not compete with billion-parameter models in raw capability.

That's not the goal.

The goal is to have a **complete, understandable, trainable LLM that you can actually run, modify, and learn from on your own hardware.**
