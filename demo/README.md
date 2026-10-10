# Web demo

A small web page where someone uploads or records a Hindi clip and sees whether the detector thinks it is a real person or AI-generated speech. It runs on Streamlit Community Cloud's free tier, on CPU.

- `streamlit_app.py`: the page.
- `detector.py`: loads the model and scores a clip. It uses frozen XLS-R features (`facebook/wav2vec2-xls-r-300m`) and the small classifier trained in notebook 03 or 05.
- `xlsr_head_clean.pt` (or `xlsr_head_b.pt` / `xlsr_head_c.pt`): the trained classifier, saved to Google Drive by the notebooks. The most robust one present is used.
- `requirements.txt` here and `packages.txt` at the repository root (ffmpeg, to read voice notes) are installed by Streamlit Community Cloud.

## Deploy

1. Put the classifier file in this folder.
2. On [share.streamlit.io](https://share.streamlit.io), sign in with GitHub and create an app from this repository: branch `main`, main file path `demo/streamlit_app.py`.

The classifier was trained on IndicSynth, which is licensed CC BY-NC 4.0, so the demo is for non-commercial use.
