"""Core of the demo: load the detector and score a clip. Shared by the Streamlit and Gradio apps.

Model: frozen XLS-R (facebook/wav2vec2-xls-r-300m) features + a small trained classifier head,
exactly as trained in notebooks 03 and 05.
"""
import os
import subprocess
import tempfile

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

CLIP_LEN = 64600          # about 4 seconds at 16 kHz: the window the detector was trained on
MAX_WINDOWS = 4           # look at up to the first 16 seconds
MIN_SECONDS = 1.0
HERE = os.path.dirname(os.path.abspath(__file__))
# The most robust head that is present is used.
HEAD_FILES = {"xlsr_head_c.pt": "robust: trained with noise, phone audio and extra fakes",
              "xlsr_head_b.pt": "robust: trained with noise and phone audio",
              "xlsr_head_clean.pt": "trained on clean audio"}


class Head(nn.Module):
    """Weighs the speech model's layers, then draws one line between real and fake."""

    def __init__(self, n_layers, dim):
        super().__init__()
        self.layer_weights = nn.Parameter(torch.zeros(n_layers))
        self.drop = nn.Dropout(0.2)
        self.out = nn.Linear(dim, 1)

    def forward(self, f):
        w = torch.softmax(self.layer_weights, 0)[None, :, None]
        return self.out(self.drop((w * f).sum(1))).squeeze(-1)


class ClipError(ValueError):
    """A problem with the uploaded clip, with a message meant for the person using the app."""


class Detector:
    def __init__(self, folder=HERE):
        from transformers import Wav2Vec2Model
        self.head_file = next((f for f in HEAD_FILES if os.path.exists(os.path.join(folder, f))), None)
        if self.head_file is None:
            raise FileNotFoundError(f"Put one of {list(HEAD_FILES)} in {folder}.")
        self.description = HEAD_FILES[self.head_file]
        ck = torch.load(os.path.join(folder, self.head_file), map_location="cpu")
        self.mu, self.sd, self.threshold = ck["mu"].float(), ck["sd"].float(), float(ck["threshold"])
        self.head = Head(self.mu.shape[1], self.mu.shape[2])
        self.head.load_state_dict(ck["head"])
        self.head.eval()
        torch.set_num_threads(os.cpu_count() or 2)
        self.ssl = Wav2Vec2Model.from_pretrained(ck["ssl_name"]).eval()

    @staticmethod
    def decode(path):
        """Any common audio file (wav, mp3, m4a, WhatsApp .opus ...) to 16 kHz mono."""
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "x.wav")
            try:
                r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", path, "-ac", "1", "-ar", "16000",
                                    "-c:a", "pcm_s16le", out], capture_output=True, text=True)
                ok = r.returncode == 0
            except FileNotFoundError:                     # no ffmpeg: fall back to soundfile
                ok = False
            try:
                if ok:
                    a, _ = sf.read(out, dtype="float32")
                    return a
                a, sr = sf.read(path, dtype="float32", always_2d=True)
            except Exception:
                raise ClipError("Could not read this audio file. Try a WAV, MP3, M4A or OGG/Opus file.")
        a = a.mean(axis=1)
        if sr != 16000:
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(sr, 16000)
            a = resample_poly(a, 16000 // g, sr // g).astype("float32")
        return a

    @staticmethod
    def windows(a):
        """Cut into 4-second windows. Short pieces are repeated to fill a window, as in training."""
        out = []
        for start in range(0, len(a), CLIP_LEN):
            w = a[start:start + CLIP_LEN]
            if len(w) < CLIP_LEN:
                if out and len(w) < 2 * 16000:
                    break
                w = np.tile(w, CLIP_LEN // len(w) + 1)[:CLIP_LEN]
            out.append(w)
            if len(out) == MAX_WINDOWS:
                break
        return np.stack(out)

    def check(self, path):
        """Returns a dict with the verdict and the numbers behind it."""
        a = self.decode(path)
        seconds = len(a) / 16000
        if seconds < MIN_SECONDS:
            raise ClipError("The clip is too short. Use at least 1 second of speech, ideally 4 or more.")
        if np.sqrt(np.mean(a ** 2)) < 1e-4:
            raise ClipError("The clip seems to be silent.")
        x = self.windows(a)
        x = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-7)
        with torch.no_grad():
            layers = self.ssl(torch.from_numpy(x), output_hidden_states=True).hidden_states
            f = torch.stack([h.mean(1) for h in layers], 1)
            logits = self.head((f - self.mu) / self.sd).numpy()
        margins = logits - self.threshold
        margin = float(margins.mean())
        return {"is_real": margin >= 0, "uncertain": abs(margin) < 1,
                "real_score": float(1 / (1 + np.exp(-margin))), "margin": margin,
                "window_margins": [float(m) for m in margins], "seconds": seconds,
                "seconds_used": min(seconds, len(x) * CLIP_LEN / 16000), "windows": len(x)}


ABOUT = """
Upload or record a short clip of someone speaking **Hindi**. The detector says whether it sounds like a real person or AI-generated speech.

**This is a research prototype, not a tool for real decisions.** Do not rely on it to judge a real call or voice note.

- **How it was tested:** on clips from speakers it never saw, it scored 0.0% equal error rate on clean audio and 0.7% or lower on voice-note and phone-call audio. A standard English-trained detector scored 73.4% on the same clips.
- **Known weak spot:** speech from one text-to-speech system it never saw (MMS-TTS) got past it 30% to 61% of the time. Other cloning tools it hasn't seen may also fool it.
- **Language:** trained on Hindi only. Other languages are untested.

Only use recordings of yourself, or of people who agreed.
"""
REPO = "https://github.com/DTOX10011/indic-voice-clone-detector"
