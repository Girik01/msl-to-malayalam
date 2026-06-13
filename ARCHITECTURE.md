# Architecture Documentation
## MSL to Malayalam — Sign Language Translation System

---

## 1. System Overview

The system translates Malayalam Sign Language (MSL) gestures into Malayalam text through an intermediate gloss representation. The pipeline is split into four independent modules:

```
┌─────────────────────────────────────────────────────────────┐
│                     Full Pipeline                           │
│                                                             │
│  Webcam                                                     │
│    │                                                        │
│    ▼  MediaPipe HandLandmarker                              │
│  Landmarks (21 keypoints × xyz = 63 features/frame)        │
│    │                                                        │
│    ▼  ┌─────────────────────────────────┐                  │
│       │     Visual Encoder Module        │                  │
│       │  FrameEncoder   [eq. 4]          │                  │
│       │  TemporalEncoder [eq. 5-6]       │                  │
│       └─────────────────────────────────┘                  │
│    │                                                        │
│    ▼  ┌─────────────────────────────────┐                  │
│       │    Gloss Generation Module       │                  │
│       │  GlossHead  [eq. 7]              │                  │
│       │  G = {g₁, g₂, …, gM}            │                  │
│       └─────────────────────────────────┘                  │
│    │                                                        │
│    ▼  ┌─────────────────────────────────┐                  │
│       │        CSAM Module               │                  │
│       │  G′ = f_align(G)  [eq. 8-9]     │                  │
│       └─────────────────────────────────┘                  │
│    │                                                        │
│    ▼  ┌─────────────────────────────────┐                  │
│       │     Translation Module           │                  │
│       │  MalayalamDecoder  [sec III-G]   │                  │
│       │  Y = {y₁, y₂, …, yN}            │                  │
│       └─────────────────────────────────┘                  │
│    │                                                        │
│    ▼  Malayalam Output                                      │
└─────────────────────────────────────────────────────────────┘
```

The intermediate gloss representation G separates the visual recognition problem from the linguistic translation problem, making the system modular and easier to improve independently at each stage.

---

## 2. Module 1 — Visual Feature Extraction (Equations 4–5)

### 2.1 Input Representation

The input is a sequence of video frames:

```
V = {f₁, f₂, …, fT}
```

where T is the total number of frames in the observation window.

Each frame fₜ is processed by **MediaPipe HandLandmarker**, which detects 21 hand keypoints and returns their (x, y, z) coordinates — 63 scalar values per frame. This gives a domain-invariant representation: the same 21-point skeleton is detected whether the hand appears in a photo, video clip, or live webcam feed.

### 2.2 Frame Encoder (Equation 4)

Each frame's landmark vector is projected into a d-dimensional feature embedding:

```
xₜ = φ(fₜ)     →     xₜ ∈ ℝᵈ
```

**Implementation (`src/model.py`, `FrameEncoder`):**

```python
class FrameEncoder(nn.Module):
    def __init__(self, in_dim=63, d=128, dropout=0.1):
        self.proj = nn.Sequential(
            nn.Linear(in_dim, d),   # xₜ = W · landmark_t + b
            nn.LayerNorm(d),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
    def forward(self, x):       # (B, T, 63) → (B, T, 128)
        return self.proj(x)
```

This produces the sequence:

```
X = {x₁, x₂, …, xT}     where each xₜ ∈ ℝ¹²⁸
```

> **Note:** The architecture specifies MobileNetV3 (`φ_MBV3`) for this projection. This implementation uses MediaPipe landmarks + linear layer instead. The interface is identical — both produce a compact xₜ ∈ ℝᵈ per frame. Landmarks are preferred here because MediaPipe's skeleton detector is domain-invariant (works on real and controlled images equally), while MobileNetV3 CNN features are sensitive to appearance differences.

---

## 3. Module 2 — Temporal Encoding (Equation 6)

Once individual frame features X are available, the temporal encoder attends across the entire sequence simultaneously to capture motion dynamics:

```
H = {h₁, h₂, …, hT} = TransformerEncoder(X)
```

Unlike recurrent networks (LSTM, GRU) which process frames one at a time, the transformer observes the complete sequence in parallel. Self-attention lets each frame hₜ incorporate information from any other frame in the sequence — important for signs that involve multi-frame trajectories.

**Implementation (`src/model.py`, `TemporalEncoder`):**

```python
class TemporalEncoder(nn.Module):
    def __init__(self, d=128, nhead=4, layers=2, dropout=0.1):
        self.pe  = PositionalEncoding(d, dropout=dropout)   # inject frame order
        enc = nn.TransformerEncoderLayer(d, nhead, d*4, dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, num_layers=layers)

    def forward(self, x, mask=None):    # (B, T, 128) → H: (B, T, 128)
        return self.enc(self.pe(x), src_key_padding_mask=mask)
```

**Parameters:**
- 2 encoder layers
- 4 attention heads
- Feedforward dim: 512 (= 4 × hidden dim 128)
- Sinusoidal positional encoding (frame order preserved)

Each vector hₜ in H encodes both the visual content of frame t and its contextual relationship with surrounding frames.

---

## 4. Module 3 — Gloss Prediction (Equation 7)

Given the temporal representations H, the model predicts which sign gesture is being performed:

```
P(g | H) = softmax(W · h̄ + b)
```

where h̄ is the mean-pooled representation of H across the time dimension. Mean pooling aggregates the full temporal context into a single vector before classification.

**Implementation (`src/model.py`, `GlossHead`):**

```python
class GlossHead(nn.Module):
    def __init__(self, d=128, n_classes=61):
        self.fc = nn.Linear(d, n_classes)

    def forward(self, H, mask=None):
        if mask is not None:
            H = H.masked_fill(mask.unsqueeze(-1), 0.0)
            lengths = (~mask).sum(1, keepdim=True).float().clamp(min=1)
            pooled  = H.sum(1) / lengths         # masked mean pool
        else:
            pooled = H.mean(1)
        return self.fc(pooled)    # → logits (B, 61)
```

**Vocabulary:** 61 MSL alphabet classes from the Sahaayi dataset.

**Training:** Stage 1 trains the FrameEncoder + TemporalEncoder + GlossHead together using cross-entropy loss against Sahaayi image labels.

**Inference:** At runtime, the top-1 prediction (if confidence ≥ 35%) is added to the gloss buffer G. The HUD displays the top-3 predictions with confidence scores.

---

## 5. Module 4 — Cross-Lingual Syntax Alignment Module / CSAM (Equations 8–9)

### 5.1 Problem: Structural Mismatch

Gloss sequences follow simplified English ordering (Subject-Verb-Object). Malayalam is a morphologically rich SOV (Subject-Object-Verb) language. Direct gloss-to-Malayalam mapping produces grammatically incorrect sentences.

Example:
```
Gloss (English SVO):   I  GO  STORE  YESTERDAY
Malayalam (SOV):       ഞാൻ  ഇന്നലെ  കടയിലേക്ക്  പോയി
                       I    yesterday  store      went
```

### 5.2 CSAM Design

The CSAM learns contextual relationships between gloss tokens using self-attention before translation. It models position-wise reordering between gloss tokens and target syntax — not as fixed rules, but as learned attention weights.

```
G = {g₁, g₂, …, gM}         (predicted gloss sequence)
G′ = f_align(G)               (aligned representation for decoder)
```

**Implementation (`src/model.py`, `CSAM`):**

```python
class CSAM(nn.Module):
    def __init__(self, d=128, n_gloss=61, nhead=4, dropout=0.1):
        self.embed = nn.Embedding(n_gloss + 3, d)   # gloss token → dense vector
        self.pe    = PositionalEncoding(d, dropout=dropout)
        self.attn  = nn.MultiheadAttention(d, nhead, dropout=dropout, batch_first=True)
        self.norm  = nn.LayerNorm(d)

    def forward(self, gloss_ids, mask=None):    # (B, M) → G′: (B, M, d)
        x = self.pe(self.embed(gloss_ids))
        a, _ = self.attn(x, x, x, key_padding_mask=mask)   # self-attention = f_align
        return self.norm(x + a)                             # residual + norm
```

The multi-head self-attention computes weighted relationships between all pairs of gloss tokens simultaneously. During training, it learns which reorderings of the gloss sequence lead to better Malayalam output — for example, discovering that `YESTERDAY` should attend heavily to `GO` to trigger the correct Malayalam verb conjugation.

---

## 6. Module 5 — Malayalam Decoder (Section III-G)

### 6.1 Autoregressive Generation

The decoder generates Malayalam text character by character:

```
Y = {y₁, y₂, …, yN}
```

At each step t, it uses:
1. **Self-attention** over previously generated characters y₁…yₜ₋₁ — maintains sentence continuity and grammatical consistency
2. **Cross-attention** over G′ from CSAM — refers back to the aligned gloss to preserve semantic meaning

This combined attention strategy allows the system to generate translations that are both semantically accurate and structurally compatible with Malayalam grammar.

**Implementation (`src/model.py`, `MalayalamDecoder`):**

```python
class MalayalamDecoder(nn.Module):
    def __init__(self, d=128, nhead=4, layers=2, n_mal=136, dropout=0.1):
        self.embed = nn.Embedding(n_mal, d, padding_idx=0)
        self.pe    = PositionalEncoding(d, dropout=dropout)
        dec = nn.TransformerDecoderLayer(d, nhead, d*4, dropout, batch_first=True)
        self.dec   = nn.TransformerDecoder(dec, num_layers=layers)
        self.out   = nn.Linear(d, n_mal)

    def forward(self, tgt, memory, tgt_mask=None, tgt_pad_mask=None):
        x = self.pe(self.embed(tgt))
        return self.out(
            self.dec(x, memory,           # memory = G′ (cross-attention source)
                     tgt_mask=tgt_mask,   # causal mask (autoregressive)
                     tgt_key_padding_mask=tgt_pad_mask)
        )
```

`nn.TransformerDecoderLayer` contains exactly both attention mechanisms:
- Masked self-attention over the partial output sequence
- Cross-attention over `memory` (= G′ from CSAM)

### 6.2 Vocabulary

Character-level Malayalam Unicode vocabulary covering the full U+0D00–U+0D7F block:

```python
MAL_CHARS  = [chr(c) for c in range(0x0D00, 0x0D80) if chr(c).strip()] + [" "]
MAL_TOKENS = ["<pad>", "<bos>", "<eos>"] + MAL_CHARS    # 136 tokens total
```

Character-level vocabulary is used because:
- Malayalam is morphologically rich — word-level vocabulary would require thousands of entries
- Any valid Malayalam string can be generated from the Unicode character set
- No out-of-vocabulary problem at the character level

### 6.3 Greedy Decoding (Inference)

```python
@torch.no_grad()
def translate(self, gloss_ids, bos, eos, max_len=80):
    G_prime = self.csam(gloss_ids.unsqueeze(0))          # G → G′
    out     = torch.tensor([[bos]])
    for _ in range(max_len):
        causal = self.causal_mask(out.size(1), out.device)
        logits = self.decoder(out, G_prime, tgt_mask=causal)
        nxt    = logits[:, -1].argmax(-1, keepdim=True)  # greedy: highest prob token
        out    = torch.cat([out, nxt], dim=1)
        if nxt.item() == eos: break
    return out[0, 1:].tolist()
```

---

## 7. Two-Stage Training

The two-stage design keeps the modules independent and allows the sign recognition encoder to be trained on available labeled data before the translation decoder is added.

### Stage 1 — Sign Recognition

| Item | Detail |
|---|---|
| Modules trained | FrameEncoder, TemporalEncoder, GlossHead |
| Dataset | Sahaayi MSL Alphabet (61 classes, ~12,200 images) |
| Loss | CrossEntropy |
| Epochs | 20 |
| Val accuracy | ~99% |

```
Landmarks → FrameEncoder → TemporalEncoder → GlossHead → class label
```

### Stage 2 — Malayalam Translation

| Item | Detail |
|---|---|
| Modules trained | CSAM, MalayalamDecoder |
| Encoder | Frozen (Stage 1 weights preserved) |
| Dataset | 31 pseudo-parallel gloss→Malayalam pairs |
| Loss | CrossEntropy + label smoothing 0.1 |
| Epochs | 50 |
| Training mode | Teacher forcing |

```
Gloss IDs → CSAM → G′ → Decoder (teacher-forced) → Malayalam characters
```

**Label smoothing (0.1)** regularizes the small training set by preventing the model from becoming overconfident on a single token prediction — important when the corpus is only 31 sentence pairs.

---

## 8. Pseudo-Parallel Data

No manually annotated gloss→Malayalam dataset exists. To train Stage 2, pseudo-parallel pairs are generated:

1. Write natural English sentences corresponding to sequences of gloss tokens
2. Machine-translate each English sentence to Malayalam using Google Translate

```python
PAIRS = [
    (["Na", "Aa", "Na"],   "I"),
    (["Va", "Aa"],         "Come"),
    (["Ka", "Zha", "I", "Ka", "Ka", "U"], "Eat"),
    ...   # 31 pairs total
]
```

The Sahaayi class names (e.g. `Na`, `Aa`, `Ka`) serve as the gloss tokens. This strategy allows Stage 2 training without any human annotation of parallel data.

---

## 9. Inference Pipeline Detail

```
Camera frame (30fps)
        │
        ▼
MediaPipe HandLandmarker
  → 21 keypoints × (x,y,z) = 63 floats
  → If no hand: clear rolling buffer immediately (prevents false predictions)
        │
        ▼
Rolling window (20 frames)
  → Every 6 frames: run model.forward_encoder()
  → Top-1 prediction if confidence ≥ 35%
        │
        ▼
GlossBuffer
  → Accept prediction if different from last sign OR > 4 s gap
  → Min 1.2 s between consecutive signs (debounce)
        │
        ▼  (after 3.5 s of silence with non-empty buffer)
model.translate(gloss_ids, bos, eos)
  → CSAM: gloss_ids → G′
  → Decoder: greedy decode G′ → Malayalam characters
        │
        ▼
Display on HUD
```

---

## 10. Design Summary

| Component | Design Choice | Reason |
|---|---|---|
| Frame features | MediaPipe landmarks | Domain-invariant; works on real and controlled images |
| Temporal model | TransformerEncoder | Captures long-range dependencies across frames; no vanishing gradient |
| Gloss vocabulary | 61 MSL alphabet classes | Only available labeled MSL data (Sahaayi) |
| Alignment module | CSAM self-attention | Learns SOV→SOV reordering from data, not fixed rules |
| Decoder | TransformerDecoder (cross-attention on G′) | Maintains semantic fidelity to gloss while generating fluent Malayalam |
| Malayalam vocab | Character-level Unicode | Handles morphological richness; no OOV problem |
| Training data S2 | Pseudo-parallel (31 pairs) | No annotated gloss→Malayalam corpus exists; scalable with more data |

---

## 11. Equation Reference

| Equation | Formula | Implemented in |
|---|---|---|
| (1) | V = {f₁, f₂, …, fT} | `inference.py` — `SignPredictor.buf` |
| (2) | V → G → Y | `model.py` — `forward_encoder`, `forward_decoder` |
| (3) | Y = {y₁, …, yN} | `train.py` — `mal_encode`, `mal_decode` |
| (4) | xₜ = φ(fₜ) | `model.py` — `FrameEncoder.forward` |
| (5) | X = {x₁, …, xT} | `model.py` — output of `FrameEncoder` |
| (6) | H = TransformerEncoder(X) | `model.py` — `TemporalEncoder.forward` |
| (7) | P(g\|H) = softmax(Wh̄+b) | `model.py` — `GlossHead.forward` |
| (8) | G = {g₁, …, gM} | `inference.py` — `GlossBuffer.words` |
| (9) | G′ = f_align(G) | `model.py` — `CSAM.forward` |
| III-G | Autoregressive decoder | `model.py` — `MalayalamDecoder.forward`, `translate` |

---

## 12. Future Scope

### 12.1 Continuous Sentence Translation

The current inference pipeline uses a single-tier gloss buffer — signs accumulate and the full buffer is translated after a silence gap. A two-tier buffer would enable real-time sentence-level translation:

```
Sign predictions
      │
      ▼ short pause (~1.5 s)
  Word committed  →  word buffer
      │
      ▼ long pause (~4 s)
  Sentence complete  →  model.translate()  →  Malayalam
```

- **Letter buffer** — accumulates individual alphabet signs into a word during fingerspelling
- **Word buffer** — accumulates committed words into a full sentence
- **Translation trigger** — fires when no new word is added for ~4 seconds

No changes to the model architecture are required. The transformer decoder already handles variable-length gloss sequences. Only the buffering logic in `inference.py` needs to be extended.

### 12.2 Word-Level MSL Vocabulary

The Sahaayi dataset covers the MSL alphabet (61 fingerspelling classes). Adding a word-level MSL dataset would allow the GlossHead to predict complete word signs directly as gloss tokens — removing the need for fingerspelling and producing more natural sentence flow. The `n_gloss` parameter in `SignToMalayalam` is configurable; the architecture scales to any vocabulary size.

### 12.3 Multi-Signer Training Data

Sahaayi contains images from a single signer under controlled conditions. Collecting data from multiple signers with varied lighting, backgrounds, and hand sizes would improve generalization. The landmark-based feature representation already provides some robustness (skeleton is person-invariant), but classifier boundaries learned from one signer may not transfer perfectly.

### 12.4 Larger Pseudo-Parallel Corpus

Stage 2 translation quality is directly proportional to the number of gloss→Malayalam training pairs. The current 31 pairs cover basic phrases. Expanding to several hundred pairs — or collecting manually annotated gloss→Malayalam sentence data — would produce significantly more fluent Malayalam output without any change to the decoder architecture.

### 12.5 Formal Evaluation Metrics

Quantitative evaluation using standard NLP metrics:

| Metric | What it measures |
|---|---|
| BLEU | N-gram overlap between generated and reference Malayalam |
| ROUGE | Recall-oriented overlap for translation completeness |
| WER (Word Error Rate) | Word-level accuracy of the generated sentence |

These metrics would allow direct comparison with CNN-LSTM baselines and other sign language translation systems.
