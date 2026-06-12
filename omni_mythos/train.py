"""train.py — OmniMythosDense, single GPU, cold start."""
import os, json, time, math
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from torch.utils.checkpoint import checkpoint
from transformers import AutoTokenizer
from safetensors.torch import save_file, load_file

from modeling_mythos import OmniMythosDense, MythosConfig

# ---------------- knobs ----------------
SEQ_LEN      = 1024
BATCH        = 1
GRAD_ACCUM   = 1
LOOPS        = 2
LR           = 1e-4
WARMUP       = 100
TOKEN_BUDGET = 40_000_000_000
CKPT_EVERY   = 500
LOG_EVERY    = 1
OUT_DIR      = "runs/v3"
RESUME       = ""            # path to a step*.safetensors to continue from a crash
TOKENIZER    = "mistralai/Mistral-7B-v0.1"

# ----------------------------------------

torch.backends.cuda.matmul.allow_tf32 = True
dev = "cuda"
os.makedirs(OUT_DIR, exist_ok=True)


class TokenStream(IterableDataset):
    def __init__(self, tok, seq_len):
        self.tok = tok
        self.seq_len = seq_len

    def __iter__(self):
        from datasets import load_dataset, interleave_datasets

        mix = [
            ("HuggingFaceTB/finemath",                       "finemath-4plus", None,     0.10),
            ("bigcode/starcoderdata",                        None,             "python", 0.15),
            ("nvidia/Nemotron-Pretraining-Specialized-v1",   "Nemotron-Pretraining-InfiniByte-Reasoning", None, 0.10),
            ("nvidia/Nemotron-Pretraining-Specialized-v1.1", "Nemotron-Pretraining-Code-Concepts",        None, 0.10),
            ("nvidia/Nemotron-Pretraining-Specialized-v1.2", "Nemotron-Pretraining-Fact-Seeking",         None, 0.10),
            ("HuggingFaceFW/fineweb",                        "sample-100BT", None, 0.25),
            ("HuggingFaceFW/fineweb-edu",                    "sample-100BT",   None,     0.20),
        ]
        probs = [w for *_, w in mix]
        s = sum(probs)
        probs = [p / s for p in probs]

        def normalize(ex):
            return {"text": ex.get("text") or ex.get("content") or ex.get("code") or ""}

        streams = []
        for name, config, data_dir, _ in mix:
            print(f"Loading {name} ...")
            ds = load_dataset(name, name=config, data_dir=data_dir,
                              split="train", streaming=True)
            ds = ds.map(normalize, remove_columns=[c for c in (ds.column_names or []) if c != "text"])
            streams.append(ds)

        print("Interleaving...")
        ds = interleave_datasets(streams, probabilities=probs, seed=0,
                                 stopping_strategy="all_exhausted")

        eos = self.tok.eos_token_id
        buf = []
        need = self.seq_len + 1
        for row in ds:
            text = (row.get("text") or "")[:500_000]
            buf.extend(self.tok.encode(text, add_special_tokens=False))
            if eos is not None:
                buf.append(eos)          # document boundary signal
            while len(buf) >= need:
                chunk, buf = buf[:need], buf[need:]
                ids = torch.tensor(chunk, dtype=torch.long)
                yield ids[:-1], ids[1:]


def main():
    # ---------------- tokenizer first: vocab_size drives the model ----------------
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    cfg = MythosConfig()
    cfg.vocab_size = tok.vocab_size
    print(f"tokenizer vocab: {tok.vocab_size}  eos: {tok.eos_token_id}")

    # ---------------- model ----------------
    model = OmniMythosDense(cfg).to(dev, dtype=torch.bfloat16)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    if RESUME and os.path.exists(RESUME):
        sd = load_file(RESUME)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"resumed from {RESUME}: {len(missing)} missing, {len(unexpected)} unexpected")
        # restore step counter from json
        json_path = RESUME.replace('.safetensors', '.json')
        if os.path.exists(json_path):
            meta = json.load(open(json_path))
            step = meta['step']
            tokens = meta['tokens']

    # activation checkpointing
    def wrap_block(blk):
        fwd = blk.forward
        blk.forward = lambda *a, **k: checkpoint(fwd, *a, **k, use_reentrant=False)

    for b in list(model.prelude_attn) + list(model.coda_attn):
        wrap_block(b)
    
   # freeze multimodal — no signal during text pretraining
    for name, param in model.named_parameters():
        if any(x in name for x in ['audio_encoder', 'vision_encoder', 'audio_head', 'image_head', 'audio_decoder', 'vision_decoder']):
            param.requires_grad = False
    print(f"trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}M")

    # optimizer: 8-bit if bnb's CUDA build is alive, fp32-state AdamW otherwise
    try:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1)
        print("optimizer: bnb AdamW8bit")
    except Exception as e:
        opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95),
                                weight_decay=0.1, fused=True)
        print(f"optimizer: torch AdamW (bnb unavailable: {e}) — drop SEQ_LEN to 512 if OOM")

    # ---------------- data (num_workers=0: no subprocess, no forkserver crash) ----
    loader = DataLoader(TokenStream(tok, SEQ_LEN), batch_size=BATCH, num_workers=0)

    def lr_at(step):
        if step < WARMUP:
            return LR * step / WARMUP
        total = TOKEN_BUDGET // (BATCH * SEQ_LEN * GRAD_ACCUM)
        prog = min(1.0, (step - WARMUP) / max(1, total - WARMUP))
        return 0.1 * LR + 0.45 * LR * (1 + math.cos(math.pi * prog))

    def save_ckpt(step):
        sd = {k: v.detach().to(torch.bfloat16).cpu().contiguous()
              for k, v in model.state_dict().items()
              if not k.startswith("lm_head")}        # tied to embed
        save_file(sd, f"{OUT_DIR}/step{step}.safetensors")
        json.dump({"step": step, "tokens": step * BATCH * SEQ_LEN * GRAD_ACCUM},
                  open(f"{OUT_DIR}/step{step}.json", "w"))
        print(f"saved step{step}")

    # ---------------- loop ----------------
    model.train()
    step = locals().get("step", 0)
    tokens = locals().get("tokens", 0)
    loss_acc, t0 = 0.0, time.time()
    t_log = time.time()
    it = iter(loader)

    while tokens < TOKEN_BUDGET:
        opt.zero_grad(set_to_none=True)
        for _ in range(GRAD_ACCUM):
            x, y = next(it)
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            logits, _, _ = model(x, n_loops=LOOPS)
            loss = F.cross_entropy(logits.float().view(-1, cfg.vocab_size), y.reshape(-1))
            (loss / GRAD_ACCUM).backward()
            loss_acc += loss.item() / GRAD_ACCUM
            tokens += BATCH * SEQ_LEN

        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gnorm):
            print(f"step {step}: non-finite grad norm, skipping update")
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
                  f"{tps:,.0f} tok/s  {tokens/1e6:.2f}M tok  "
                  f"vram {torch.cuda.max_memory_allocated()/1e9:.2f}GB")
            loss_acc = 0.0
            t_log = time.time()

        if step % CKPT_EVERY == 0:
            save_ckpt(step)

    save_ckpt(step)


if __name__ == "__main__":
    main()
