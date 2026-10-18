"""Streamlit entry point for CNN image classification."""
import hashlib

import streamlit as st

from src.model_loader import ModelError, load_config, load_model_cached, model_file
from src.preprocessing import ImageError, preprocess_image, validate_image
from src.predictor import predict
from src.ui_components import header, inject_css, result_card, status_badge


st.set_page_config(page_title="AI Image Classification", page_icon="🧠", layout="wide")
inject_css()
header()
st.caption("Model CIFAR-10 mengenali 10 kelas: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, dan truck.")

try:
    config = load_config()
except ModelError as exc:
    status_badge("Configuration Error", "error")
    st.error(str(exc))
    st.stop()

available = model_file(config).is_file()
status_area = st.empty()
with status_area.container():
    status_badge("Ready" if available else "Model Not Available", "ready" if available else "error")
if not available:
    st.info(f"Tambahkan model CNN kelompok ke `{config['model_path']}`, lalu sesuaikan `config/model_config.json` dan muat ulang halaman.")

st.markdown('<div class="section-title">Upload gambar</div>', unsafe_allow_html=True)
uploaded = st.file_uploader("Pilih file JPG, JPEG, atau PNG", type=["jpg", "jpeg", "png"], help=f"Ukuran maksimum {config['max_upload_mb']:g} MB.")

image = None
upload_key = None
if uploaded is not None:
    try:
        image = validate_image(uploaded, config["max_upload_mb"])
        upload_key = hashlib.sha256(uploaded.getvalue()).hexdigest()
        left, right = st.columns([1.2, 1], gap="large")
        with left:
            st.image(image, caption=f"Preview · {uploaded.name}", use_container_width=True)
        with right:
            st.markdown('<div class="section-title">Detail gambar</div>', unsafe_allow_html=True)
            st.write(f"**Nama file:** {uploaded.name}")
            st.write(f"**Dimensi asli:** {image.width} × {image.height} px")
            st.write(f"**Ukuran:** {len(uploaded.getvalue()) / (1024 * 1024):.2f} MB")
            st.write(f"**Input model:** {config['image_size'][1]} × {config['image_size'][0]} px")
    except ImageError as exc:
        st.error(str(exc))

if st.session_state.get("upload_key") != upload_key:
    st.session_state.pop("prediction", None)
    st.session_state["upload_key"] = upload_key

analyze = st.button("Analyze Image", type="primary", use_container_width=True, disabled=image is None or not available)
if analyze and image is not None:
    try:
        with status_area.container():
            status_badge("Loading")
        with st.spinner("Loading model dan menganalisis gambar..."):
            model = load_model_cached(config)
            batch = preprocess_image(image, config)
            st.session_state["prediction"] = predict(model, batch, config)
        with status_area.container():
            status_badge("Ready", "ready")
    except (ModelError, ImageError) as exc:
        st.session_state.pop("prediction", None)
        with status_area.container():
            status_badge("Model Not Available", "error")
        st.error(str(exc))
    except Exception as exc:
        st.session_state.pop("prediction", None)
        with status_area.container():
            status_badge("Model Not Available", "error")
        st.error(f"Analisis gagal karena kesalahan tak terduga: {exc}")

if st.session_state.get("prediction") is not None and image is not None:
    st.markdown('<div class="section-title">Hasil analisis</div>', unsafe_allow_html=True)
    result_card(st.session_state["prediction"])
