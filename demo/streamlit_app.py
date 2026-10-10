"""Hindi voice-clone detector: web demo for Streamlit Community Cloud.

Deploy: share.streamlit.io → Create app → this repository, branch main, file demo/streamlit_app.py.
"""
import os
import sys
import tempfile

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detector import ABOUT, REPO, ClipError, Detector  # noqa: E402

st.set_page_config(page_title="Hindi voice-clone detector")


@st.cache_resource(show_spinner="Loading the speech model (about a minute the first time)...")
def load():
    return Detector()


st.title("Hindi voice-clone detector")
st.markdown(ABOUT)
detector = load()
st.caption(f"Model in use: {detector.description}. Clips are deleted as soon as they are checked. "
           f"[Code, data and full results on GitHub]({REPO})")

upload_tab, record_tab = st.tabs(["Upload a clip", "Record"])
with upload_tab:
    uploaded = st.file_uploader("Hindi speech clip",
                                type=["wav", "mp3", "m4a", "ogg", "opus", "flac", "aac", "webm"])
with record_tab:
    recorded = st.audio_input("Record yourself speaking Hindi")
clip = uploaded or recorded

if st.button("Check this voice", type="primary", disabled=clip is None):
    suffix = os.path.splitext(getattr(clip, "name", "") or "")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(clip.getvalue())
        path = f.name
    try:
        with st.spinner("Listening..."):
            r = detector.check(path)
    except ClipError as e:
        st.error(str(e))
    else:
        if r["is_real"]:
            st.success("### Sounds like a real person")
        else:
            st.error("### Sounds AI-generated")
        if r["uncertain"]:
            st.warning("The score is close to the line, so treat this as uncertain.")
        st.progress(r["real_score"], text=f"Leans real: {100 * r['real_score']:.0f}% "
                                          "(a score, not a calibrated probability)")
        st.caption(f"Looked at the first {r['seconds_used']:.0f} of {r['seconds']:.0f} seconds, in "
                   f"{r['windows']} window(s) of about 4 seconds. Window scores: "
                   + ", ".join(f"{m:+.1f}" for m in r["window_margins"])
                   + " (above 0 leans real, below 0 leans AI-generated).")
    finally:
        os.remove(path)
