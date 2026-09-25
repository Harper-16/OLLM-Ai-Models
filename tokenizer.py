from pathlib import Path
from tokenizers import Tokenizer as HFTokenizer


class Tokenizer:
    """AE-model BPE tokenizer."""

    def __init__(self, path=None):
        if path is None:
            raise ValueError("Tokenizer path is required")

        path = Path(path).expanduser()

        if not path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {path}")

        self.kind = "bpe"
        self.path = str(path)

        self.tok = HFTokenizer.from_file(self.path)
        self.vocab_size = self.tok.get_vocab_size()

        vocab = self.tok.get_vocab()

        self.pad_id = vocab.get("<PAD>", 0)
        self.unk_id = vocab.get("<UNK>", 1)
        self.bos_id = vocab.get("<BOS>")
        self.eos_id = vocab.get("<EOS>")

        if self.vocab_size != 8192:
            raise ValueError(
                f"Expected 8192-token BPE tokenizer, "
                f"but found {self.vocab_size}"
            )

    def encode(self, text):
        """Convert text -> token IDs."""
        return self.tok.encode(str(text)).ids

    def decode(self, ids):
        """Convert token IDs -> text."""
        return self.tok.decode([int(x) for x in ids])

    def save_info(self, path):
        import json

        info = {
            "kind": self.kind,
            "path": self.path,
            "vocab_size": self.vocab_size,
            "pad_id": self.pad_id,
            "unk_id": self.unk_id,
            "bos_id": self.bos_id,
            "eos_id": self.eos_id,
        }

        Path(path).write_text(
            json.dumps(info, indent=2),
            encoding="utf-8",
        )


# ------------------------------------------------------------
# Direct test
# ------------------------------------------------------------

if __name__ == "__main__":
    import os

    default_path = os.path.expanduser(
        "~/Model-Az/AE-model/data/bpe_tokenizer.json"
    )

    t = Tokenizer(default_path)

    tests = [
        "Hello, this is my artificial intelligence model.",
        "The future of artificial intelligence is fascinating.",
        "Transformers use self attention to understand context.",
    ]

    print("=" * 64)
    print("AE-MODEL BPE TOKENIZER")
    print("=" * 64)
    print("kind      =", t.kind)
    print("vocab     =", t.vocab_size)
    print("PAD       =", t.pad_id)
    print("UNK       =", t.unk_id)
    print("BOS       =", t.bos_id)
    print("EOS       =", t.eos_id)
    print()

    for text in tests:
        ids = t.encode(text)
        decoded = t.decode(ids)

        print("TEXT   :", text)
        print("TOKENS :", len(ids))
        print("IDS    :", ids)
        print("DECODE :", decoded)
        print()

        if not decoded.strip():
            raise RuntimeError("Tokenizer produced empty decoded text")

    print("=" * 64)
    print("BPE TOKENIZER: PASS")
    print("=" * 64)
