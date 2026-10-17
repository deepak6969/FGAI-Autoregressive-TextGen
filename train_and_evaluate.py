"""Autoregressive word-level text generation: SimpleRNN vs BiRNN vs LSTM vs GRU.

Usage:
    python train_and_evaluate.py --out results            # full run
    python train_and_evaluate.py --out smoke --quick      # 1-epoch smoke test
"""
import argparse
import copy
import json
import math
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import textgen_data as D

MODELS = {"SimpleRNN": "rnn", "BiRNN": "birnn", "LSTM": "lstm", "GRU": "gru"}
SEEDS = [
    "to sherlock holmes she is always the woman",
    "i had seen little of holmes lately",
    "the case had been an interesting one and",
]
TEMPS = [0.5, 1.0, 1.3]
SAMPLE_SEEDS = [0, 1, 2]


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


class LM(nn.Module):
    """Shared embedding + recurrent backbone + linear head; only the backbone changes."""

    def __init__(self, kind, vocab, emb, hid, layers, drop):
        super().__init__()
        self.bi = kind == "birnn"
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.drop = nn.Dropout(drop)
        kw = dict(input_size=emb, hidden_size=hid, num_layers=layers, batch_first=True,
                  dropout=drop if layers > 1 else 0.0)
        if kind in ("rnn", "birnn"):
            self.rnn = nn.RNN(nonlinearity="tanh", bidirectional=self.bi, **kw)
        elif kind == "lstm":
            self.rnn = nn.LSTM(**kw)
        else:
            self.rnn = nn.GRU(**kw)
        self.fc = nn.Linear(hid * (2 if self.bi else 1), vocab)

    def forward(self, x):
        _, h = self.rnn(self.drop(self.emb(x)))
        if isinstance(h, tuple):          # LSTM returns (h, c)
            h = h[0]
        last = torch.cat([h[-2], h[-1]], 1) if self.bi else h[-1]
        return self.fc(self.drop(last))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    crit = nn.CrossEntropyLoss(reduction="sum")
    loss = c1 = c5 = n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss += crit(out, y).item()
        top5 = out.topk(5, 1).indices
        c1 += (top5[:, 0] == y).sum().item()
        c5 += (top5 == y[:, None]).any(1).sum().item()
        n += len(y)
    return loss / n, c1 / n, c5 / n


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def train_model(name, kind, data, cfg, vocab, device):
    set_seed(cfg["seed"])
    model = LM(kind, vocab, cfg["emb"], cfg["hid"], cfg["layers"], cfg["dropout"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    crit = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(cfg["seed"])
    tr = torch.utils.data.DataLoader(data["train"], cfg["batch"], shuffle=True, generator=g)
    va = torch.utils.data.DataLoader(data["val"], 1024)
    hist, best, best_state, train_time = [], float("inf"), None, 0.0
    for ep in range(1, cfg["epochs"] + 1):
        model.train(); sync(device); t0 = time.time(); tot = cnt = 0
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
            opt.step()
            tot += loss.item() * len(y); cnt += len(y)
        sync(device); dt = time.time() - t0; train_time += dt
        vl, v1, v5 = evaluate(model, va, device)
        hist.append(dict(model=name, epoch=ep, train_loss=tot / cnt, val_loss=vl,
                         val_ppl=math.exp(vl), val_acc=v1, val_top5=v5, epoch_time=dt))
        print(f"[{name}] ep{ep} train={tot/cnt:.3f} val={vl:.3f} ppl={math.exp(vl):.1f} "
              f"acc={v1:.3f} ({dt:.1f}s)", flush=True)
        if vl < best:
            best, best_state = vl, copy.deepcopy(model.state_dict())
            best_ep = ep
    model.load_state_dict(best_state)
    te = torch.utils.data.DataLoader(data["test"], 1024)
    tl, t1, t5 = evaluate(model, te, device)
    res = dict(model=name, params=n_params, best_epoch=best_ep, train_time_s=train_time,
               val_loss=best, test_loss=tl, test_ppl=math.exp(tl), test_acc=t1, test_top5=t5)
    return model, hist, res


@torch.no_grad()
def generate(model, seed_ids, n_words, temp, seed, length, device):
    model.eval()
    gen = torch.Generator(device=device).manual_seed(seed)
    ids = list(seed_ids)
    for _ in range(n_words):
        ctx = ids[-length:]
        ctx = [0] * (length - len(ctx)) + ctx          # left-pad short seeds with <pad>
        logits = model(torch.tensor([ctx], device=device))[0]
        logits[0] = logits[1] = float("-inf")           # never emit <pad>/<unk>
        probs = torch.softmax(logits / temp, -1)
        ids.append(torch.multinomial(probs, 1, generator=gen).item())
    return ids[len(seed_ids):]


def md_table(df, floatfmt="{:.3f}"):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, float) else str(v) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--data", default="data/pg1661.txt")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--quick", action="store_true", help="1 epoch, for a smoke test")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    cfg = dict(vocab=5000, length=20, stride=1, emb=128, hid=256, layers=2, dropout=0.3,
               lr=1e-3, batch=256, clip=1.0, epochs=1 if a.quick else a.epochs, seed=42,
               gen_words=40)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # ---- data -------------------------------------------------------------
    tokens = D.tokenize(D.load_text(a.data))
    n = len(tokens)
    tr_t = tokens[: int(n * 0.8)]
    itos, stoi = D.build_vocab(tr_t, cfg["vocab"])
    ids = D.encode(tokens, stoi)
    parts = dict(zip(["train", "val", "test"], D.split_stream(ids)))
    data, stats = {}, dict(dataset=D.DATASET_NAME, url=D.DATASET_URL, total_tokens=n,
                           vocab_size=len(itos), unique_raw_words=len(set(tokens)))
    for k, v in parts.items():
        x, y = D.make_windows(v, cfg["length"], cfg["stride"])
        data[k] = torch.utils.data.TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
        stats[k] = dict(tokens=len(v), windows=len(y), unk_rate=float((v == 1).mean()))
    print(json.dumps(stats, indent=2))
    json.dump(stats, open(f"{a.out}/dataset_statistics.json", "w"), indent=2)
    json.dump(cfg, open(f"{a.out}/config.json", "w"), indent=2)

    # ---- train + evaluate -------------------------------------------------
    results, history, samples, trained = [], [], [], {}
    for name, kind in MODELS.items():
        model, hist, res = train_model(name, kind, data, cfg, len(itos), device)
        results.append(res); history += hist; trained[name] = model

    # ---- generation -------------------------------------------------------
    for name, model in trained.items():
        for seed_text in SEEDS:
            seed_ids = [stoi.get(w, 1) for w in D.tokenize(seed_text)]
            for T in TEMPS:
                for s in SAMPLE_SEEDS:
                    out = generate(model, seed_ids, cfg["gen_words"], T, s, cfg["length"], device)
                    words = [itos[i] for i in out]
                    samples.append(dict(
                        model=name, seed=seed_text, temperature=T, sample_seed=s,
                        distinct1=D.distinct_n(words, 1), distinct2=D.distinct_n(words, 2),
                        repeated_trigram=D.repeated_trigram_fraction(words),
                        longest_run=D.longest_repeat_run(words), text=" ".join(words)))

    res_df, hist_df, samp_df = map(pd.DataFrame, (results, history, samples))
    res_df.to_csv(f"{a.out}/results.csv", index=False)
    hist_df.to_csv(f"{a.out}/history.csv", index=False)
    samp_df.to_csv(f"{a.out}/samples.csv", index=False)
    temp_df = (samp_df.groupby(["model", "temperature"])
               [["distinct1", "distinct2", "repeated_trigram", "longest_run"]]
               .mean().reset_index())
    temp_df.to_csv(f"{a.out}/temperature_summary.csv", index=False)

    # ---- plots ------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for m, d in hist_df.groupby("model"):
        ax[0].plot(d.epoch, d.val_loss, label=m)
        ax[1].plot(d.epoch, d.val_ppl, label=m)
        ax[2].plot(d.epoch, d.val_acc * 100, label=m)
    for x_, t in zip(ax, ["Validation loss", "Validation perplexity", "Validation accuracy (%)"]):
        x_.set_title(t); x_.set_xlabel("epoch"); x_.legend()
    plt.tight_layout(); plt.savefig(f"{a.out}/learning_curves.png", dpi=150); plt.close()

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for x_, col, t in zip(ax, ["distinct1", "distinct2", "repeated_trigram"],
                          ["Distinct-1", "Distinct-2", "Repeated-trigram fraction"]):
        for m, d in temp_df.groupby("model"):
            x_.plot(d.temperature, d[col], marker="o", label=m)
        x_.set_title(t); x_.set_xlabel("temperature"); x_.legend()
    plt.tight_layout(); plt.savefig(f"{a.out}/temperature_metrics.png", dpi=150); plt.close()

    # ---- ready-to-paste README tables ------------------------------------
    with open(f"{a.out}/results_tables.md", "w") as f:
        f.write("### Test results\n\n" + md_table(res_df) + "\n\n")
        f.write("### Temperature summary (mean over 3 seeds x 3 sampling seeds)\n\n"
                + md_table(temp_df) + "\n\n### Example generations (sample_seed=0)\n\n")
        for _, r in samp_df[samp_df.sample_seed == 0].iterrows():
            f.write(f"- **{r.model} | T={r.temperature} | \"{r.seed}\"** -> {r.text}\n")
    print("\n" + md_table(res_df))
    print(f"\nAll outputs saved to {a.out}/")


if __name__ == "__main__":
    main()
