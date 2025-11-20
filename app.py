import streamlit as st
from src.infer import predict

st.title("Fake/Real Image Detector")
up = st.file_uploader("Upload image", type=["png","jpg","jpeg"])
if up:
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
        f.write(up.read())
        path = f.name
    label, conf = predict(path)
    st.image(path, caption=f"{label} ({conf:.2f})")
