from pathlib import Path
import random
import torch
from torch.utils.data import IterableDataset

class StreamingTextDataset(IterableDataset):
    """
    Streams .txt files and never loads the whole corpus into RAM.
    It builds a token buffer and emits fixed-length next-token examples.
    """
    def __init__(self, roots, tokenizer, seq_len=256, shuffle_files=True, seed=1337):
        super().__init__()
        self.roots = [Path(x).expanduser() for x in roots]
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.shuffle_files = shuffle_files
        self.seed = seed
        self.files = []
        for root in self.roots:
            if root.is_file():
                self.files.append(root)
            elif root.is_dir():
                self.files.extend(root.rglob("*.txt"))
        self.files = sorted(set(self.files))
        if not self.files:
            raise FileNotFoundError("No .txt files found in the supplied data roots.")

    def __iter__(self):
        files = list(self.files)
        rng = random.Random(self.seed + getattr(self, "_epoch", 0))
        if self.shuffle_files:
            rng.shuffle(files)

        # Each worker gets a disjoint slice of files.
        info = torch.utils.data.get_worker_info()
        if info is not None:
            files = files[info.id::info.num_workers]

        buffer = []
        for path in files:
            try:
                with path.open("r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        ids = self.tokenizer.encode(line)
                        if self.tokenizer.eos_id is not None:
                            ids.append(self.tokenizer.eos_id)
                        buffer.extend(ids)
                        while len(buffer) >= self.seq_len + 1:
                            chunk = buffer[:self.seq_len + 1]
                            del buffer[:self.seq_len + 1]
                            x = torch.tensor(chunk[:-1], dtype=torch.long)
                            y = torch.tensor(chunk[1:], dtype=torch.long)
                            yield x, y
            except OSError as e:
                print(f"[dataset] skipping {path}: {e}")
