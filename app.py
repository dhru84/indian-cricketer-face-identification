"""Streamlit demo.  Run: streamlit run app.py   (after `python -m src.train`)"""
import streamlit as st
from PIL import Image

from src.config import MODEL_PATH

st.set_page_config(page_title="Cricketer Face ID", page_icon="🏏", layout="wide")
st.title("🏏 Indian Cricket Team – Face Identification")
st.caption("Detects faces, identifies trained players, and says **Unknown** for anyone else.")

if not MODEL_PATH.exists():
    st.error("No trained model found. Add images to data/raw/<Player>/ and run `python -m src.train` first.")
    st.stop()


@st.cache_resource
def load():
    from src.predict import FaceIdentifier
    return FaceIdentifier()


identifier = load()
threshold = st.sidebar.slider("Minimum confidence threshold", 0.2, 0.95, float(identifier.threshold), 0.05,
                              help="Higher = stricter: more faces are labelled Unknown.")
st.sidebar.write("**Players:** " + ", ".join(map(str, identifier.classes)))
st.sidebar.info("For education only. Do not use to identify private individuals without consent.")

file = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png", "webp"])
if file:
    img = Image.open(file)
    identifier.threshold = threshold
    results = identifier.identify(img)
    left, right = st.columns([3, 2])
    left.image(identifier.annotate(img, results), use_container_width=True)
    with right:
        if not results:
            st.warning("No face detected.")
        for i, r in enumerate(results, 1):
            st.subheader(f"Face {i}: {r['name']}")
            for name, p in r["top_k"]:
                st.write(f"{name}")
                st.progress(min(p, 1.0), text=f"{p:.1%}")
