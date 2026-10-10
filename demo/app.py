"""Hindi voice-clone detector: a small web demo.

Upload or record a clip; the app says whether it sounds like a real person or AI-generated speech.
Model: frozen XLS-R (facebook/wav2vec2-xls-r-300m) features + a small trained classifier head.
Code and results: https://github.com/DTOX10011/indic-voice-clone-detector
"""
import os
import subprocess
import tempfile

import gradio as gr
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

CLIP_LEN = 64600          # about 4 seconds at 16 kHz: the window the detector was trained on
MAX_WINDOWS = 4           # look at up to the first 16 seconds
MIN_SECONDS = 1.0
# The most robust head that has been uploaded is used.
HEAD_FILES = ["xlsr_head_c.pt", "xlsr_head_b.pt", "xlsr_head_clean.pt"]
HEAD_NAMES = {"xlsr_head_c.pt": "robust (noise, phone audio and extra fakes)",
              "xlsr_head_b.pt": "robust (noise and phone audio)",
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


HERE = os.path.dirname(os.path.abspath(__file__))
head_file = next((f for f in HEAD_FILES if os.path.exists(os.path.join(HERE, f))), None)
if head_file is None:
    raise FileNotFoundError(f"Upload one of {HEAD_FILES} next to app.py.")
ck = torch.load(os.path.join(HERE, head_file), map_location="cpu")
mu, sd, threshold = ck["mu"].float(), ck["sd"].float(), float(ck["threshold"])
head = Head(mu.shape[1], mu.shape[2])
head.load_state_dict(ck["head"])
head.eval()
torch.set_num_threads(os.cpu_count() or 2)
ssl = Wav2Vec2Model.from_pretrained(ck["ssl_name"]).eval()


def decode(path):
    """Any audio file (wav, mp3, m4a, WhatsApp .opus ...) to 16 kHz mono."""
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "x.wav")
        r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", path, "-ac", "1", "-ar", "16000",
                            "-c:a", "pcm_s16le", out], capture_output=True, text=True)
        if r.returncode != 0:
            raise gr.Error("Could not read this audio file. Try a WAV, MP3, M4A or OGG/Opus file.")
        a, _ = sf.read(out, dtype="float32")
    return a


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


def score(a):
    x = windows(a)
    x = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-7)
    with torch.no_grad():
        layers = ssl(torch.from_numpy(x), output_hidden_states=True).hidden_states
        f = torch.stack([h.mean(1) for h in layers], 1)
        return head((f - mu) / sd).numpy(), len(x)


def analyze(path):
    if not path:
        raise gr.Error("Upload or record a clip first.")
    a = decode(path)
    seconds = len(a) / 16000
    if seconds < MIN_SECONDS:
        raise gr.Error("The clip is too short. Use at least 1 second of speech, ideally 4 or more.")
    if np.sqrt(np.mean(a ** 2)) < 1e-4:
        raise gr.Error("The clip seems to be silent.")
    logits, n = score(a)
    margin = float(logits.mean() - threshold)
    real = 1 / (1 + np.exp(-margin))
    if margin >= 0:
        verdict = "## Sounds like a real person"
    else:
        verdict = "## Sounds AI-generated"
    if abs(margin) < 1:
        verdict += "\n\nThe score is close to the line, so treat this as uncertain."
    used = min(seconds, n * CLIP_LEN / 16000)
    details = (f"Looked at the first {used:.0f} of {seconds:.0f} seconds, in {n} window(s) of about 4 seconds. "
               f"Window scores: {', '.join(f'{v - threshold:+.1f}' for v in logits)} "
               f"(above 0 leans real, below 0 leans AI-generated).")
    return verdict, {"Real": real, "AI-generated": 1 - real}, details


ABOUT = f"""
# Hindi voice-clone detector

Upload or record a short clip of someone speaking **Hindi**. The detector says whether it sounds like a real person or AI-generated speech.

**This is a research prototype, not a tool for real decisions.** Do not rely on it to judge a real call or voice note.

- **How it was tested:** on clips from speakers it never saw, it scored 0.0% equal error rate on clean audio and 0.7% or lower on voice-note and phone-call audio. A standard English-trained detector scored 73.4% on the same clips.
- **Known weak spot:** speech from one text-to-speech system it never saw (MMS-TTS) got past it 30% to 61% of the time. Other cloning tools it hasn't seen may also fool it.
- **Language:** trained on Hindi only. Other languages are untested.
- **Model in use:** {HEAD_NAMES[head_file]}.

Uploaded clips are deleted from the server within about an hour. Only use recordings of yourself, or of people who agreed.

[Code, data and full results on GitHub](https://github.com/DTOX10011/indic-voice-clone-detector)
"""

with gr.Blocks(title="Hindi voice-clone detector", delete_cache=(3600, 3600)) as demo:
    gr.Markdown(ABOUT)
    with gr.Row():
        with gr.Column():
            audio = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Hindi speech clip")
            go = gr.Button("Check this voice", variant="primary")
        with gr.Column():
            verdict = gr.Markdown()
            scores = gr.Label(num_top_classes=2, label="Detector score (a score, not a calibrated probability)")
            details = gr.Markdown()
    go.click(analyze, inputs=audio, outputs=[verdict, scores, details])

if __name__ == "__main__":
    demo.launch()
