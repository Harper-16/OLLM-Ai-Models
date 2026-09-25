"""
Small supervised-fine-tuning stage.

Input JSONL format:
{"instruction":"Explain gravity simply.","response":"Gravity is ..."}
{"instruction":"Write a Python loop.","response":"for i in range(...):"}

This keeps the same causal LM and teaches instruction -> response behavior.
"""
import argparse, json, os
from pathlib import Path
import torch
from model import GPTModel
from tokenizer import Tokenizer
from checkpoint import load_checkpoint

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokenizer", default="data/sharded_model/tokenizer.json")
    p.add_argument("--out", default="runs/sft.pt")
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--layers", type=int, default=12)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--kv-heads", type=int, default=2)
    p.add_argument("--ffn", type=int, default=768)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=5e-5)
    args=p.parse_args()

    torch.set_num_threads(max(1,min(8,os.cpu_count() or 1)))
    tok=Tokenizer(args.tokenizer)
    model=GPTModel(tok.vocab_size,args.seq_len,args.dim,args.layers,args.heads,args.kv_heads,args.ffn)
    load_checkpoint(args.checkpoint, model, map_location="cpu")
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,betas=(0.9,0.95),weight_decay=0.01)
    rows=[json.loads(x) for x in Path(args.data).open(encoding="utf-8") if x.strip()]
    model.train()
    for step in range(args.steps):
        row=rows[step % len(rows)]
        text=f"User: {row['instruction']}\nAssistant: {row['response']}"
        ids=tok.encode(text)[-args.seq_len:]
        if len(ids)<2: continue
        x=torch.tensor([ids[:-1]],dtype=torch.long)
        y=torch.tensor([ids[1:]],dtype=torch.long)
        _,loss=model(x,y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        if step%50==0: print(f"step={step} loss={loss.item():.4f}")
    torch.save({"model":model.state_dict(),"config":vars(args),"step":args.steps},args.out)
    print("saved",args.out)

if __name__=="__main__": main()
