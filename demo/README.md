---
title: Hindi Voice-Clone Detector
emoji: 🎙️
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
pinned: false
license: cc-by-nc-4.0
short_description: Real or AI-generated? A research prototype for Hindi
---

# Hindi voice-clone detector

A research prototype that says whether a Hindi speech clip sounds like a real person or AI-generated speech. It uses frozen XLS-R features (`facebook/wav2vec2-xls-r-300m`) and a small classifier trained on Hindi real speech (Kathbath) and fakes (IndicSynth).

Not a tool for real decisions. Results, limits and code: https://github.com/DTOX10011/indic-voice-clone-detector

The classifier was trained on IndicSynth, which is licensed CC BY-NC 4.0, so this demo is for non-commercial use.
