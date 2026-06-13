# MSL to Malayalam — Sign Language Translation

A transformer-based system that translates Malayalam Sign Language (MSL) gestures into Malayalam text in real time.

```
Webcam → Hand Landmarks → Transformer Encoder → Gloss → CSAM → Malayalam
```

---

## Quick Start (No Training Required)

Pre-trained checkpoints are included. Just install dependencies and run.

```bash
pip install torch torchvision mediapipe opencv-python deep-translator numpy
```

```bash
cd src
python inference.py
```

Press **SPACE** to clear the sign buffer. Press **Q** to quit.

---

## Project Structure

```
msl-to-malayalam/
│
├── src/
│   ├── model.py            # Transformer architecture (FrameEncoder, CSAM, Decoder)
│   ├── train.py            # Two-stage training pipeline
│   ├── pseudo_data.py      # Generate pseudo-parallel gloss→Malayalam pairs
│   └── inference.py        # Live webcam demo
│
├── checkpoints/
│   ├── landmarks_stage1.npz   # Pre-extracted MediaPipe landmarks (Sahaayi dataset)
│   ├── pseudo_parallel.json   # Gloss→Malayalam training pairs
│   ├── model_stage1.pt        # Stage 1 checkpoint (encoder + gloss head)
│   └── model_stage2.pt        # Stage 2 checkpoint (full model with decoder)
│
├── poc/
│   └── hand_landmarker.task   # MediaPipe hand landmark model
│
├── Sahaayi---Model-Creation-master/
│   └── image_data/            # MSL alphabet dataset — 61 classes, ~200 images each
│
├── README.md
└── ARCHITECTURE.md            # Detailed architecture documentation
```

---

## Setup

### Requirements

- Python 3.9–3.13
- Apple Silicon (MPS), CUDA, or CPU

### 1. Install dependencies

```bash
pip install torch torchvision mediapipe opencv-python deep-translator numpy
```

### 2. Apple Silicon — set this environment variable

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

Add it to your shell profile (`~/.zshrc`) to make it permanent.

### 3. MediaPipe hand model

The file `poc/hand_landmarker.task` is included. If missing:

```bash
curl -L https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task \
     -o poc/hand_landmarker.task
```

---

## Running

### Live inference (pre-trained model)

```bash
cd src
python inference.py
```

The HUD displays:
- **Hand** — whether a hand is currently detected
- **Top-3** — top 3 predicted sign classes with confidence scores
- **Gloss (G)** — accumulated sign buffer
- **CSAM (G′)** — syntax alignment step (paper module)
- **Malayalam** — translated output, generated after 3.5 s of silence

### Controls

| Key | Action |
|---|---|
| `SPACE` | Clear sign buffer and reset |
| `Q` | Quit |

---

## Training From Scratch

Training is optional — checkpoints are provided.

### Step 1 — Generate pseudo-parallel data

```bash
cd src
python pseudo_data.py
```

Creates `checkpoints/pseudo_parallel.json` — gloss→Malayalam sentence pairs.

### Step 2 — Train

```bash
cd src
python train.py           # runs both stages
python train.py --stage 1 # Stage 1 only: encoder + gloss head
python train.py --stage 2 # Stage 2 only: CSAM + decoder
```

**Stage 1** (~20 epochs, 2–5 min): trains the sign recognition encoder on Sahaayi MSL alphabet images.

**Stage 2** (~50 epochs, fast): freezes the encoder and trains the CSAM alignment module and Malayalam decoder on pseudo-parallel data.

---

## Dataset

### Sahaayi MSL Alphabet (`Sahaayi---Model-Creation-master/image_data/`)

- 61 Malayalam Sign Language alphabet classes
- ~200 real hand photographs per class (~12,200 total)
- MediaPipe landmark detection rate: **100%** (real photographs)
- Stage 1 validation accuracy: **~99%**

### Why only Sahaayi?

Other available datasets (ISL animated videos) use 3D avatar/cartoon hands. MediaPipe was trained on real human hands and detects landmarks unreliably on animated hands. Sahaayi provides real photographs, giving clean and reliable landmark features.

---

## Performance

| Stage | Task | Accuracy |
|---|---|---|
| Stage 1 | MSL alphabet recognition (61 classes) | ~99% val accuracy |
| Stage 2 | Gloss → Malayalam translation | Functional; limited by pseudo-parallel data size |

Stage 1 accuracy proves the architecture works correctly. Stage 2 translation quality is bounded by the 31 pseudo-parallel sentence pairs used for training — it will improve proportionally with more annotated gloss→Malayalam data.

---

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full technical documentation including equation-by-equation alignment with the research design.

---

## Known Limitations

1. **Single signer in training data** — Sahaayi images are from one person; accuracy may vary across signers
2. **Small translation corpus** — 31 pseudo-parallel pairs; decoder output is limited to trained phrase patterns
3. **MSL alphabet only** — word-level MSL signs are not included (no public dataset exists)
4. **Continuous signing** — system recognizes one sign at a time with pause-based segmentation
