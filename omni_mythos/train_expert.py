"""train_expert.py — Train a domain-specialist expert backbone and push to HuggingFace.

Usage:
    EXPERT=edu python train_expert.py
    EXPERT=mathcode python train_expert.py
    EXPERT=reasoning python train_expert.py
    EXPERT=factual python train_expert.py
"""

import os, json, time, math
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoTokenizer
from safetensors.torch import save_file, load_file
from huggingface_hub import HfApi, upload_file
import tempfile

from expert_model import ExpertLM, ExpertConfig

# ---------------- knobs ----------------
EXPERT       = os.environ.get("EXPERT", "edu")   # edu | mathcode | reasoning | factual
SEQ_LEN      = 1024
BATCH        = 80
GRAD_ACCUM   = 48
LR           = 3e-4
WARMUP       = 200
TOKEN_BUDGET = 10_000_000_000   # 10B tokens
CKPT_EVERY   = 1000
LOG_EVERY    = 1
HF_REPO      = f"Johnblick187/OmniMythos-Expert-{EXPERT}"
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
TOKENIZER    = "mistralai/Mistral-7B-v0.1"
RESUME       = os.environ.get("RESUME", "")
# ----------------------------------------

torch.backends.cuda.matmul.allow_tf32 = True
dev = "cuda"

# Data mix per expert
DATA_MIXES = {
    "edu": [
        ("HuggingFaceFW/fineweb-edu", "sample-100BT", None, 0.70),
        ("HuggingFaceFW/fineweb",     "sample-100BT", None, 0.30),
    ],
    "mathcode": [
        ("HuggingFaceFW/fineweb-edu",  "sample-100BT",   None,     0.20),
        ("HuggingFaceTB/finemath",     "finemath-4plus",  None,     0.50),
        ("bigcode/starcoderdata",       None,              "python", 0.30),
    ],
    "reasoning": [
        ("HuggingFaceFW/fineweb-edu",  "sample-100BT", None, 0.20),
        ("nvidia/Nemotron-Pretraining-Specialized-v1",   "Nemotron-Pretraining-InfiniByte-Reasoning", None, 0.50),
        ("nvidia/Nemotron-Pretraining-Specialized-v1.1", "Nemotron-Pretraining-Code-Concepts",        None, 0.30),
    ],
    "factual": [
        ("HuggingFaceFW/fineweb-edu",  "sample-100BT", None, 0.30),
        ("nvidia/Nemotron-Pretraining-Specialized-v1.2", "Nemotron-Pretraining-Fact-Seeking", None, 0.70),
    ],
}


class TokenStream(IterableDataset):
    def __init__(self, tok, seq_len, mix):
        self.tok = tok
        self.seq_len = seq_len
        self.mix = mix

    def __iter__(self):
        from datasets import load_dataset, interleave_datasets

        probs = [w for *_, w in self.mix]
        s = sum(probs)
        probs = [p / s for p in probs]

        def normalize(ex):
            return {"text": ex.get("text") or ex.get("content") or ex.get("code") or ""}

        streams = []
        for name, config, data_dir, _ in self.mix:
            print(f"Loading {name} ...")
            ds = load_dataset(name, name=config, data_dir=data_dir,
                              split="train", streaming=True)
            ds = ds.map(normalize, remove_columns=[c for c in (ds.column_names or []) if c != "text"])
            streams.append(ds)

        print("Interleaving...")
        ds = interleave_datasets(streams, probabilities=probs, seed=42,
                                 stopping_strategy="all_exhausted")

        eos = self.tok.eos_token_id
        buf = []
        need = self.seq_len + 1
        for row in ds:
            text = (row.get("text") or "")[:500_000]
            buf.extend(self.tok.encode(text, add_special_tokens=False))
            if eos is not None:
                buf.append(eos)
            while len(buf) >= need:
                chunk, buf = buf[:need], buf[need:]
                ids = torch.tensor(chunk, dtype=torch.long)
                yield ids[:-1], ids[1:]


def push_to_hf(path, repo_id, token, filename):
    if not token:
        print("No HF_TOKEN set, skipping HF push")
        return
    try:
        api = HfApi()
        api.create_repo(repo_id=repo_id, token=token, exist_ok=True, private=False)
        upload_file(
            path_or_fileobj=path,
            path_in_repo=filename,
            repo_id=repo_id,
            token=token,
        )
        print(f"Pushed {filename} to {repo_id}")
    except Exception as e:
        print(f"HF push failed: {e}")


def main():
    print(f"Training expert: {EXPERT}")
    mix = DATA_MIXES[EXPERT]

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    cfg = ExpertConfig()
    cfg.vocab_size = tok.vocab_size

    model = ExpertLM(cfg).to(dev, dtype=torch.bfloat16)
    total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"params: {total:.1f}M")

    step   = 0
    tokens = 0

    if RESUME and os.path.exists(RESUME):
        sd = load_file(RESUME)
        model.load_state_dict(sd, strict=False)
        json_path = RESUME.replace('.safetensors', '.json')
        if os.path.exists(json_path):
            meta = json.load(open(json_path))
            step   = meta['step']
            tokens = meta['tokens']
            print(f"resumed step={step} tokens={tokens/1e9:.2f}B")

    try:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1)
        print("optimizer: bnb AdamW8bit")
    except Exception as e:
        opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95),
                                weight_decay=0.1, fused=True)
        print(f"optimizer: torch AdamW ({e})")

    loader = DataLoader(TokenStream(tok, SEQ_LEN, mix), batch_size=BATCH,
                        num_workers=4, pin_memory=True, prefetch_factor=2)

    def lr_at(step):
        if step < WARMUP:
            return LR * step / WARMUP
        total_steps = TOKEN_BUDGET // (BATCH * SEQ_LEN * GRAD_ACCUM)
        prog = min(1.0, (step - WARMUP) / max(1, total_steps - WARMUP))
        return 0.1 * LR + 0.45 * LR * (1 + math.cos(math.pi * prog))

    def save_and_push(step):
        sd = {k: v.detach().to(torch.bfloat16).cpu().contiguous()
              for k, v in model.state_dict().items()
              if not k.startswith("lm_head")}
        meta = {"step": step, "tokens": tokens, "expert": EXPERT}

        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = os.path.join(tmp, f"step{step}.safetensors")
            meta_path = os.path.join(tmp, f"step{step}.json")
            save_file(sd, ckpt_path)
            json.dump(meta, open(meta_path, "w"))
            push_to_hf(ckpt_path, HF_REPO, HF_TOKEN, f"step{step}.safetensors")
            push_to_hf(meta_path, HF_REPO, HF_TOKEN, f"step{step}.json")

        print(f"saved step{step} ({tokens/1e9:.2f}B tokens)")

    model.train()
    loss_acc = 0.0
    t_log = time.time()
    it = iter(loader)

    while tokens < TOKEN_BUDGET:
        opt.zero_grad(set_to_none=True)
        for _ in range(GRAD_ACCUM):
            x, y = next(it)
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            logits = model(x)
            loss = F.cross_entropy(logits.float().view(-1, cfg.vocab_size), y.reshape(-1))
            (loss / GRAD_ACCUM).backward()
            loss_acc += loss.item() / GRAD_ACCUM
            tokens += BATCH * SEQ_LEN

        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gnorm):
            print(f"step {step}: non-finite grad norm, skipping")
            opt.zero_grad(set_to_none=True)
            step += 1
            continue

        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.step()
        step += 1

        if step % LOG_EVERY == 0:
            dt = time.time() - t_log
            tps = (LOG_EVERY * BATCH * SEQ_LEN * GRAD_ACCUM) / dt
            print(f"step {step}  loss {loss_acc/LOG_EVERY:.4f}  "
                  f"ppl {math.exp(min(20, loss_acc/LOG_EVERY)):.1f}  "
                  f"{tps:,.0f} tok/s  {tokens/1e9:.2f}B tok")
            loss_acc = 0.0
            t_log = time.time()

        if step % CKPT_EVERY == 0:
            save_and_push(step)

    save_and_push(step)


if __name__ == "__main__":
    main()
