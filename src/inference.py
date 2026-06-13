"""
Live webcam inference — paper pipeline

  Webcam → MediaPipe landmarks → Encoder → Gloss (G)
                                                ↓
                                        CSAM → G'
                                                ↓
                                   Decoder → Malayalam (Y)
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import cv2, time, argparse, torch, numpy as np
from collections import deque

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from train import GLOSS_CLASSES, MAL_TOKENS, PAD_IDX, BOS_IDX, EOS_IDX, mal_decode
from model import SignToMalayalam

BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LM_FILE  = os.path.join(BASE, "poc", "hand_landmarker.task")
DEVICE   = torch.device("cpu")

GLOSS2IDX = {c: i for i, c in enumerate(GLOSS_CLASSES)}


def make_landmarker():
    opts = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=LM_FILE),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=1, min_hand_detection_confidence=0.5,
    )
    return mp_vision.HandLandmarker.create_from_options(opts)


def draw_hand(frame, hl, w, h):
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hl]
    for c in mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS:
        cv2.line(frame, pts[c.start], pts[c.end], (0, 220, 0), 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, (255, 255, 255), -1)


class SignPredictor:
    """Rolling window → classify gloss sign. Clears when hand disappears."""
    def __init__(self, model, window=20, stride=6, threshold=0.35):
        self.model     = model
        self.window    = window
        self.stride    = stride
        self.threshold = threshold
        self.buf       = deque(maxlen=window)
        self.count     = 0
        self.top3      = []
        self.conf      = 0.0

    def update(self, lm_vec):
        if lm_vec is None:
            self.buf.clear(); self.top3 = []; self.conf = 0.0
            return None

        self.buf.append(lm_vec); self.count += 1
        if len(self.buf) < self.window // 3: return None
        if self.count % self.stride != 0:
            return (self.top3[0][0] if self.top3 and self.conf >= self.threshold
                    else None)

        x = torch.tensor(np.stack(self.buf), dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            _, logits = self.model.forward_encoder(x)
            probs = torch.softmax(logits, -1)[0]
            confs, preds = probs.topk(3)

        self.top3 = [(GLOSS_CLASSES[preds[i].item()], confs[i].item()) for i in range(3)]
        self.conf = confs[0].item()
        return GLOSS_CLASSES[preds[0].item()] if self.conf >= self.threshold else None


class GlossBuffer:
    def __init__(self, min_gap=1.2):
        self.words = []; self._last = None; self._t = 0; self.min_gap = min_gap

    def push(self, word):
        if not word: return False
        now = time.time()
        if word == self._last and now - self._t < 4.0: return False
        if now - self._t < self.min_gap: return False
        self.words.append(word); self._last = word; self._t = now
        return True

    def clear(self): self.words.clear(); self._last = None; self._t = 0


def run(model_path):
    ckpt  = torch.load(model_path, map_location=DEVICE)
    model = SignToMalayalam(n_gloss=len(GLOSS_CLASSES), n_mal=len(MAL_TOKENS))
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Model: {os.path.basename(model_path)}")

    predictor = SignPredictor(model)
    buf       = GlossBuffer()
    detector  = make_landmarker()
    cap       = cv2.VideoCapture(0)
    if not cap.isOpened(): print("Cannot open webcam"); return

    malayalam  = ""
    last_act   = time.time()
    GAP        = 3.5

    while True:
        ok, frame = cap.read()
        if not ok: break
        frame = cv2.flip(frame, 1); h, w = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        lm_vec = None
        if res.hand_landmarks:
            hl     = res.hand_landmarks[0]
            lm_vec = np.array([[p.x,p.y,p.z] for p in hl], dtype=np.float32).flatten()
            draw_hand(frame, hl, w, h)

        label = predictor.update(lm_vec)
        if label and buf.push(label):
            last_act = time.time(); malayalam = ""

        if buf.words and (time.time() - last_act) > GAP and not malayalam:
            ids = [GLOSS2IDX[w] for w in buf.words if w in GLOSS2IDX]
            if ids:
                g = torch.tensor(ids, device=DEVICE)
                out = model.translate(g, BOS_IDX, EOS_IDX)
                malayalam = mal_decode(out)

        # ── HUD ──────────────────────────────────────────────────────────
        ov = frame.copy()
        cv2.rectangle(ov, (0, h-200), (w, h), (10,10,10), -1)
        cv2.addWeighted(ov, 0.7, frame, 0.3, 0, frame)

        def put(txt, y, col=(230,230,230), sc=0.58):
            cv2.putText(frame, txt, (10,y), cv2.FONT_HERSHEY_SIMPLEX, sc, col, 1, cv2.LINE_AA)

        top3 = "  ".join(f"{l}({c:.0%})" for l,c in predictor.top3) if predictor.top3 else "—"
        hand = "detected" if lm_vec is not None else "no hand"
        gloss_str = " | ".join(buf.words) if buf.words else "—"

        put(f"Hand      : {hand}",               h-185, (100,255,100))
        put(f"Top-3     : {top3}",               h-155, (160,240,160), 0.50)
        put(f"Gloss (G) : {gloss_str[:60]}",     h-118, (255,220, 80))
        put(f"CSAM (G') : self-attention align",  h- 88, ( 80,200,255))
        put(f"Malayalam : {(malayalam or '—')[:55]}", h-55, (255,130,200), 0.62)
        if malayalam and len(malayalam) > 55:
            put(f"            {malayalam[55:]}",  h-25, (255,130,200), 0.62)

        put("SPACE=clear  Q=quit", 22, (150,150,150), 0.46)
        cv2.putText(frame, os.path.basename(model_path),
                    (w-300,22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (130,130,130), 1)

        # Confidence bar
        fill = int(predictor.conf * 180)
        cv2.rectangle(frame, (w-200, h-18), (w-20, h-6), (50,50,50), -1)
        cv2.rectangle(frame, (w-200, h-18), (w-200+fill, h-6),
                      (0,200,80) if predictor.conf >= 0.35 else (0,140,200), -1)

        cv2.imshow("Sign Language -> Malayalam  (paper pipeline)", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'): break
        elif k == ord(' '): buf.clear(); malayalam = ""

    cap.release(); cv2.destroyAllWindows(); detector.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    ckpt_dir = os.path.join(BASE, "checkpoints")
    default  = (os.path.join(ckpt_dir, "model_stage2.pt")
                if os.path.exists(os.path.join(ckpt_dir, "model_stage2.pt"))
                else os.path.join(ckpt_dir, "model_stage1.pt"))
    p.add_argument("--model", default=default)
    args = p.parse_args()
    if not os.path.exists(args.model):
        print("No model found. Run: python train.py"); exit(1)
    run(args.model)
