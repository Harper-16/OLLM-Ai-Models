import argparse, os
import torch
from model import GPTModel
from tokenizer import Tokenizer
from checkpoint import load_checkpoint

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="runs/ultimate/final.pt")
    p.add_argument("--tokenizer", default="data/sharded_model/tokenizer.json")
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--layers", type=int, default=12)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--kv-heads", type=int, default=2)
    p.add_argument("--ffn", type=int, default=768)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    args = p.parse_args()

    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    tok = Tokenizer(args.tokenizer)
    model = GPTModel(tok.vocab_size, args.seq_len, args.dim, args.layers,
                     args.heads, args.kv_heads, args.ffn)
    ckpt = load_checkpoint(args.checkpoint, model, map_location="cpu")
    model.eval()
    print(f"loaded step={ckpt.get('step')} params={sum(p.numel() for p in model.parameters()):,}")
    print("Type /quit to exit. This is a base next-token model unless you later SFT it.")

    while True:
        try:
            prompt = input("\nYou: ")
        except (EOFError, KeyboardInterrupt):
            break
        if prompt.strip() in {"/quit", "/exit"}:
            break
        ids = tok.encode(prompt)
        if not ids:
            continue
        x = torch.tensor([ids[-args.seq_len:]], dtype=torch.long)
        with torch.inference_mode():
            out = model.generate(x, 160, args.temperature, args.top_k, tok.eos_id)
        answer = tok.decode(out[0].tolist())
        if answer.startswith(prompt):
            answer = answer[len(prompt):]
        print("AD: " + answer)

if __name__ == "__main__":
    main()
