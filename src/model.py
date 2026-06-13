"""
Sign Language to Malayalam — Transformer Architecture (paper Figure 1)

Input: hand landmark sequence (T frames × 63 features)

  FrameEncoder      — eq(4)  x_t = projection(landmark_t)
  TemporalEncoder   — eq(5-6) H = TransformerEncoder(X)
  GlossHead         — eq(7)  P(g|H) = softmax(W · mean(H) + b)
  CSAM              — eq(8-9) G' = f_align(G)  [self-attention]
  MalayalamDecoder  — sec III-G autoregressive decoder
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d, max_len=512, dropout=0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.drop(x + self.pe[:, :x.size(1)])


# ── Equation (4): frame-level feature extraction ──────────────────────────────
class FrameEncoder(nn.Module):
    def __init__(self, in_dim=63, d=128, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, d),
            nn.LayerNorm(d),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
    def forward(self, x):          # (B,T,63) → (B,T,d)
        return self.proj(x)


# ── Equations (5-6): temporal encoding ───────────────────────────────────────
class TemporalEncoder(nn.Module):
    def __init__(self, d=128, nhead=4, layers=2, dropout=0.1):
        super().__init__()
        self.pe  = PositionalEncoding(d, dropout=dropout)
        enc = nn.TransformerEncoderLayer(d, nhead, d*4, dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, num_layers=layers)

    def forward(self, x, mask=None):   # (B,T,d) → H: (B,T,d)
        return self.enc(self.pe(x), src_key_padding_mask=mask)


# ── Equation (7): gloss prediction ────────────────────────────────────────────
class GlossHead(nn.Module):
    def __init__(self, d=128, n_classes=61):
        super().__init__()
        self.fc = nn.Linear(d, n_classes)

    def forward(self, H, mask=None):
        # Mean-pool H (ignoring padding), then classify
        if mask is not None:
            H = H.masked_fill(mask.unsqueeze(-1), 0.0)
            lengths = (~mask).sum(1, keepdim=True).float().clamp(min=1)
            pooled  = H.sum(1) / lengths
        else:
            pooled = H.mean(1)
        return self.fc(pooled)           # (B, n_classes)


# ── Equations (8-9): CSAM ─────────────────────────────────────────────────────
class CSAM(nn.Module):
    """
    Takes gloss token embeddings and learns cross-lingual reordering
    via self-attention before passing to the Malayalam decoder.
    """
    def __init__(self, d=128, n_gloss=61, nhead=4, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(n_gloss + 3, d)   # +3 for PAD/BOS/EOS
        self.pe    = PositionalEncoding(d, dropout=dropout)
        self.attn  = nn.MultiheadAttention(d, nhead, dropout=dropout, batch_first=True)
        self.norm  = nn.LayerNorm(d)

    def forward(self, gloss_ids, mask=None):    # (B,M) → G': (B,M,d)
        x = self.pe(self.embed(gloss_ids))
        a, _ = self.attn(x, x, x, key_padding_mask=mask)
        return self.norm(x + a)


# ── Section III-G: Malayalam decoder ─────────────────────────────────────────
class MalayalamDecoder(nn.Module):
    """
    Autoregressive decoder:
      - self-attention  → understands previously generated Malayalam tokens
      - cross-attention → refers back to aligned gloss representation G'
    """
    def __init__(self, d=128, nhead=4, layers=2, n_mal=136, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(n_mal, d, padding_idx=0)
        self.pe    = PositionalEncoding(d, dropout=dropout)
        dec = nn.TransformerDecoderLayer(d, nhead, d*4, dropout, batch_first=True)
        self.dec   = nn.TransformerDecoder(dec, num_layers=layers)
        self.out   = nn.Linear(d, n_mal)

    def forward(self, tgt, memory, tgt_mask=None, tgt_pad_mask=None):
        x = self.pe(self.embed(tgt))
        return self.out(self.dec(x, memory, tgt_mask=tgt_mask,
                                 tgt_key_padding_mask=tgt_pad_mask))


# ── Full model ────────────────────────────────────────────────────────────────
class SignToMalayalam(nn.Module):
    def __init__(self, n_gloss=61, n_mal=136, d=128, nhead=4,
                 enc_layers=2, dec_layers=2, dropout=0.1):
        super().__init__()
        self.frame_enc = FrameEncoder(63, d, dropout)
        self.temp_enc  = TemporalEncoder(d, nhead, enc_layers, dropout)
        self.gloss_head = GlossHead(d, n_gloss)
        self.csam       = CSAM(d, n_gloss, nhead, dropout)
        self.decoder    = MalayalamDecoder(d, nhead, dec_layers, n_mal, dropout)

    @staticmethod
    def causal_mask(sz, device):
        return torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()

    def forward_encoder(self, lm, mask=None):
        """landmarks → H and gloss logits"""
        H = self.temp_enc(self.frame_enc(lm), mask)
        return H, self.gloss_head(H, mask)

    def forward_decoder(self, gloss_ids, tgt_ids, gloss_mask=None, tgt_mask=None):
        """gloss ids + target ids → Malayalam logits (teacher-forced)"""
        G_prime = self.csam(gloss_ids, gloss_mask)
        causal  = self.causal_mask(tgt_ids.size(1), tgt_ids.device)
        return self.decoder(tgt_ids, G_prime, tgt_mask=causal,
                            tgt_pad_mask=tgt_mask)

    @torch.no_grad()
    def translate(self, gloss_ids, bos, eos, max_len=80):
        """Greedy decode gloss_ids → Malayalam token ids"""
        G_prime = self.csam(gloss_ids.unsqueeze(0))        # (1,M,d)
        out     = torch.tensor([[bos]], device=gloss_ids.device)
        for _ in range(max_len):
            causal = self.causal_mask(out.size(1), out.device)
            logits = self.decoder(out, G_prime, tgt_mask=causal)
            nxt    = logits[:, -1].argmax(-1, keepdim=True)
            out    = torch.cat([out, nxt], dim=1)
            if nxt.item() == eos:
                break
        return out[0, 1:].tolist()
