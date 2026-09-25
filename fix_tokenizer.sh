#!/usr/bin/env bash
set -e

ROOT="$HOME/Model-Az/AE-model"
DATA="$HOME/Model-Az/AC-model/data/training_shards"
OUT="$ROOT/data/bpe_tokenizer.json"

mkdir -p "$ROOT/data"

echo "============================================================"
echo "AE-model tokenizer repair"
echo "============================================================"

python3 - <<'PY'
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

data = Path.home() / "Model-Az/AC-model/data/training_shards"
out = Path.home() / "Model-Az/AE-model/data/bpe_tokenizer.json"

files = sorted(data.glob("*.txt"))

if not files:
    raise SystemExit("ERROR: no training shards found")

print("Training files:", len(files))

tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(
    vocab_size=8192,
    min_frequency=2,
    special_tokens=[
        "<PAD>",
        "<UNK>",
        "<BOS>",
        "<EOS>",
    ],
)

tokenizer.train([str(x) for x in files], trainer)

tokenizer.save(str(out))

print()
print("BPE tokenizer created")
print("path:", out)
print("vocab:", tokenizer.get_vocab_size())

for text in [
    "Hello, this is my artificial intelligence model.",
    "The future of artificial intelligence is fascinating.",
    "Transformers use self attention to understand context.",
]:
    enc = tokenizer.encode(text)
    dec = tokenizer.decode(enc.ids)

    print()
    print("TEXT :", text)
    print("TOKENS:", len(enc.ids))
    print("IDS  :", enc.ids[:30])
    print("DECODE:", dec)
PY

echo
echo "============================================================"
echo "Tokenizer created successfully"
echo "============================================================"
echo "$OUT"
