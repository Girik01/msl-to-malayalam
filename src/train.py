"""
Two-stage training following the paper.

Stage 1 — Train encoder + gloss head on Sahaayi landmarks (MSL alphabet)
Stage 2 — Train CSAM + decoder on pseudo-parallel gloss→Malayalam pairs

Usage:
  python train.py          # runs both stages
  python train.py --stage 1
  python train.py --stage 2
"""

import os, json, argparse
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from model import SignToMalayalam

BASE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT    = os.path.join(BASE, "checkpoints")
LM_NPZ  = os.path.join(CKPT, "landmarks_stage1.npz")
PP_JSON = os.path.join(CKPT, "pseudo_parallel.json")

DEVICE = torch.device("mps"  if torch.backends.mps.is_available() else
                       "cuda" if torch.cuda.is_available() else "cpu")

# ── Vocabularies ──────────────────────────────────────────────────────────────

def load_gloss_classes():
    d = np.load(LM_NPZ, allow_pickle=True)
    return d["classes"].tolist()          # 61 MSL alphabet labels

GLOSS_CLASSES = load_gloss_classes()
GLOSS2IDX     = {c: i for i, c in enumerate(GLOSS_CLASSES)}

# Malayalam character vocabulary (Unicode block U+0D00–U+0D7F + space)
MAL_CHARS  = [chr(c) for c in range(0x0D00, 0x0D80) if chr(c).strip()] + [" "]
MAL_TOKENS = ["<pad>", "<bos>", "<eos>"] + MAL_CHARS
MAL2IDX    = {t: i for i, t in enumerate(MAL_TOKENS)}
PAD_IDX    = MAL2IDX["<pad>"]
BOS_IDX    = MAL2IDX["<bos>"]
EOS_IDX    = MAL2IDX["<eos>"]

def mal_encode(text):
    ids = [BOS_IDX]
    for ch in text:
        if ch in MAL2IDX:
            ids.append(MAL2IDX[ch])
    ids.append(EOS_IDX)
    return ids

def mal_decode(ids):
    skip = {PAD_IDX, BOS_IDX, EOS_IDX}
    return "".join(MAL_TOKENS[i] for i in ids if i not in skip and i < len(MAL_TOKENS))

# ── Stage 1 dataset ───────────────────────────────────────────────────────────

class SahaayiDataset(Dataset):
    """Single landmark frames from Sahaayi MSL alphabet images."""
    def __init__(self, augment=False):
        data          = np.load(LM_NPZ, allow_pickle=True)
        self.X        = data["landmarks"].astype(np.float32)   # (N, 63)
        self.y        = data["labels"].astype(np.int64)
        self.augment  = augment

    def __len__(self): return len(self.y)

    def __getitem__(self, i):
        x = torch.tensor(self.X[i])
        if self.augment:
            x = x + torch.randn_like(x) * 0.008
        # Shape (1, 63) — single frame, T=1
        return x.unsqueeze(0), torch.tensor(self.y[i])


# ── Stage 2 dataset ───────────────────────────────────────────────────────────

class PseudoParallelDataset(Dataset):
    """Pseudo-parallel gloss→Malayalam pairs."""
    def __init__(self):
        with open(PP_JSON, encoding="utf-8") as f:
            pairs = json.load(f)
        self.samples = []
        for p in pairs:
            g_ids = [GLOSS2IDX[w] for w in p["gloss"] if w in GLOSS2IDX]
            m_ids = mal_encode(p["malayalam"])
            if g_ids and len(m_ids) > 2:
                self.samples.append((g_ids, m_ids))

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        g, m = self.samples[i]
        return torch.tensor(g, dtype=torch.long), torch.tensor(m, dtype=torch.long)


def pp_collate(batch):
    gs, ms = zip(*batch)
    B = len(gs)
    mg = max(g.size(0) for g in gs);  mm = max(m.size(0) for m in ms)
    gp = torch.full((B, mg), 0, dtype=torch.long)
    mp_ = torch.full((B, mm), PAD_IDX, dtype=torch.long)
    gm = torch.ones(B, mg, dtype=torch.bool)
    mm_ = torch.ones(B, mm, dtype=torch.bool)
    for i, (g, m) in enumerate(zip(gs, ms)):
        gp[i, :g.size(0)] = g;  gm[i, :g.size(0)] = False
        mp_[i,:m.size(0)] = m;  mm_[i,:m.size(0)] = False
    return gp, gm, mp_, mm_


# ── Stage 1 ───────────────────────────────────────────────────────────────────

def stage1(epochs=20):
    print("\n=== Stage 1: Encoder + Gloss Head (Sahaayi MSL) ===")
    ds    = SahaayiDataset(augment=True)
    n_val = max(1, int(len(ds) * 0.15))
    tr, vl = random_split(ds, [len(ds) - n_val, n_val])
    tr_l  = DataLoader(tr, batch_size=64, shuffle=True,  num_workers=0)
    vl_l  = DataLoader(vl, batch_size=64, shuffle=False, num_workers=0)
    print(f"Classes: {len(GLOSS_CLASSES)}  Train: {len(tr)}  Val: {len(vl)}")

    model = SignToMalayalam(n_gloss=len(GLOSS_CLASSES), n_mal=len(MAL_TOKENS)).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit  = nn.CrossEntropyLoss()
    best  = 0.0

    for ep in range(1, epochs + 1):
        model.train()
        tr_corr = tr_tot = 0
        for x, y in tr_l:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            _, logits = model.forward_encoder(x)
            loss = crit(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_corr += (logits.argmax(1) == y).sum().item()
            tr_tot  += y.size(0)

        model.eval()
        vl_corr = vl_tot = 0
        with torch.no_grad():
            for x, y in vl_l:
                x, y = x.to(DEVICE), y.to(DEVICE)
                _, logits = model.forward_encoder(x)
                vl_corr += (logits.argmax(1) == y).sum().item()
                vl_tot  += y.size(0)

        sch.step()
        tr_acc = tr_corr / tr_tot;  vl_acc = vl_corr / vl_tot
        print(f"Ep {ep:02d}/{epochs}  train_acc={tr_acc:.3f}  val_acc={vl_acc:.3f}")

        if vl_acc > best:
            best = vl_acc
            torch.save({"model": model.state_dict(),
                        "gloss_classes": GLOSS_CLASSES,
                        "mal_tokens": MAL_TOKENS},
                       os.path.join(CKPT, "model_stage1.pt"))
            print(f"  → saved  val_acc={vl_acc:.3f}")

    print(f"Stage 1 done. Best val acc: {best:.3f}")
    return best


# ── Stage 2 ───────────────────────────────────────────────────────────────────

def stage2(epochs=50):
    print("\n=== Stage 2: CSAM + Malayalam Decoder (pseudo-parallel) ===")

    if not os.path.exists(PP_JSON):
        print("Missing pseudo_parallel.json — generating...")
        from pseudo_data import generate
        generate()

    ds = PseudoParallelDataset()
    if len(ds) == 0:
        print("No valid pairs found. Check pseudo_parallel.json"); return

    n_val = max(1, int(len(ds) * 0.1))
    tr, vl = random_split(ds, [len(ds) - n_val, n_val])
    tr_l = DataLoader(tr, batch_size=16, shuffle=True,
                      collate_fn=pp_collate, num_workers=0)
    vl_l = DataLoader(vl, batch_size=16, shuffle=False,
                      collate_fn=pp_collate, num_workers=0)
    print(f"Pairs: {len(ds)}  Train: {len(tr)}  Val: {len(vl)}")

    model = SignToMalayalam(n_gloss=len(GLOSS_CLASSES), n_mal=len(MAL_TOKENS)).to(DEVICE)
    s1_ckpt = os.path.join(CKPT, "model_stage1.pt")
    if os.path.exists(s1_ckpt):
        model.load_state_dict(torch.load(s1_ckpt, map_location=DEVICE)["model"],
                              strict=False)
        print("  Loaded stage-1 weights.")

    # Freeze encoder — only train CSAM + decoder
    for p in model.frame_enc.parameters():  p.requires_grad = False
    for p in model.temp_enc.parameters():   p.requires_grad = False
    for p in model.gloss_head.parameters(): p.requires_grad = False

    opt  = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=5e-4, weight_decay=1e-4)
    sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1)
    best = float("inf")

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0
        for g, gm, m, mm in tr_l:
            g, gm, m = g.to(DEVICE), gm.to(DEVICE), m.to(DEVICE)
            tgt_in  = m[:, :-1];  tgt_out = m[:, 1:]
            logits  = model.forward_decoder(g, tgt_in, gloss_mask=gm)
            loss    = crit(logits.reshape(-1, len(MAL_TOKENS)), tgt_out.reshape(-1))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item()

        model.eval()
        vl_loss = 0
        with torch.no_grad():
            for g, gm, m, mm in vl_l:
                g, gm, m = g.to(DEVICE), gm.to(DEVICE), m.to(DEVICE)
                tgt_in  = m[:, :-1];  tgt_out = m[:, 1:]
                logits  = model.forward_decoder(g, tgt_in, gloss_mask=gm)
                vl_loss += crit(logits.reshape(-1, len(MAL_TOKENS)),
                                tgt_out.reshape(-1)).item()
        sch.step()

        avg_tr = tr_loss / len(tr_l);  avg_vl = vl_loss / len(vl_l)
        print(f"Ep {ep:02d}/{epochs}  train_loss={avg_tr:.4f}  val_loss={avg_vl:.4f}")
        if avg_vl < best:
            best = avg_vl
            torch.save({"model": model.state_dict(),
                        "gloss_classes": GLOSS_CLASSES,
                        "mal_tokens": MAL_TOKENS},
                       os.path.join(CKPT, "model_stage2.pt"))
            print(f"  → saved  val_loss={avg_vl:.4f}")

    print(f"Stage 2 done. Best val loss: {best:.4f}")

    # Quick sanity check
    print("\nSanity check translations:")
    model.eval()
    tests = [["Hello"], ["I", "Go", "Home"], ["You", "Come", "Now"]]
    for gloss in tests:
        ids = [GLOSS2IDX[w] for w in gloss if w in GLOSS2IDX]
        if not ids: continue
        g_t = torch.tensor(ids, device=DEVICE)
        out = model.translate(g_t, BOS_IDX, EOS_IDX)
        print(f"  {gloss} → {mal_decode(out)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all", choices=["1","2","all"])
    args = parser.parse_args()
    print(f"Device: {DEVICE}")
    if args.stage in ("1","all"): stage1()
    if args.stage in ("2","all"): stage2()
