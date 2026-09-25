import argparse, json, math, os, time
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from model import GPTModel
from tokenizer import Tokenizer
from dataset import StreamingTextDataset
from checkpoint import save_checkpoint, load_checkpoint

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", default=["data/training_shards", "data/combined_train"])
    p.add_argument("--tokenizer", default="data/sharded_model/tokenizer.json")
    p.add_argument("--out", default="runs/ultimate")
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--layers", type=int, default=12)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--kv-heads", type=int, default=2)
    p.add_argument("--ffn", type=int, default=768)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--resume", default="")
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=10)
    args = p.parse_args()

    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cpu")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tok = Tokenizer(args.tokenizer)
    print(f"tokenizer={tok.kind} vocab={tok.vocab_size}")
    model = GPTModel(
        vocab_size=tok.vocab_size,
        max_seq_len=args.seq_len,
        dim=args.dim,
        n_layers=args.layers,
        n_heads=args.heads,
        n_kv_heads=args.kv_heads,
        ffn_hidden=args.ffn,
    ).to(device)
    params = sum(x.numel() for x in model.parameters())
    print(f"parameters={params:,} ({params/1e6:.2f}M)")
    print(f"device={device} threads={torch.get_num_threads()}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                                  weight_decay=args.weight_decay)
    dataset = StreamingTextDataset(args.data, tok, seq_len=args.seq_len)
    loader = DataLoader(dataset, batch_size=args.batch, num_workers=args.workers, pin_memory=False)
    it = iter(loader)

    start_step = 0
    if args.resume:
        ckpt = load_checkpoint(args.resume, model, optimizer, map_location="cpu")
        start_step = int(ckpt.get("step", 0)) + 1
        print(f"resumed from step {start_step}")

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / max(1, args.warmup)
        progress = (step - args.warmup) / max(1, args.steps - args.warmup)
        progress = min(1.0, max(0.0, progress))
        return args.lr * 0.1 + 0.5 * (args.lr - args.lr * 0.1) * (1 + math.cos(math.pi * progress))

    model.train()
    running = 0.0
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)

    for step in range(start_step, args.steps):
        lr = lr_at(step)
        for group in optimizer.param_groups:
            group["lr"] = lr

        total_loss = 0.0
        for micro in range(args.grad_accum):
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(loader)
                x, y = next(it)
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            (loss / args.grad_accum).backward()
            total_loss += float(loss.detach())

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        running += total_loss / args.grad_accum
        if step % args.log_every == 0:
            avg = running / max(1, args.log_every)
            elapsed = time.time() - t0
            print(f"step={step:6d} loss={avg:.4f} lr={lr:.2e} tok/s~{args.batch*args.grad_accum*args.seq_len*args.log_every/max(elapsed,1e-6):.0f}")
            running = 0.0
            t0 = time.time()

        if step > start_step and step % args.save_every == 0:
            save_checkpoint(out / "latest.pt", model, optimizer, step, config=vars(args))

    save_checkpoint(out / "final.pt", model, optimizer, args.steps - 1, config=vars(args))
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))

if __name__ == "__main__":
    main()
