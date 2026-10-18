"""Presentation components for the Streamlit dashboard."""
import pandas as pd
import streamlit as st


def inject_css():
    st.markdown("""<style>
    .stApp {background: radial-gradient(circle at 85% 0%, #1a1c40 0, #0b1020 38%);}
    .block-container {max-width: 1120px; padding-top: 5rem; padding-bottom: 4rem;}
    h1, h2, h3 {letter-spacing: -.035em;}
    .eyebrow {color:#ad9fff; font-size:.78rem; font-weight:800; letter-spacing:.16em; text-transform:uppercase; margin-bottom:.5rem;}
    .hero-title {font-size:clamp(2.2rem,5vw,4rem); line-height:1.08; font-weight:800; color:#f5f6ff; letter-spacing:-.055em; margin:0 0 .8rem;}
    .hero-copy {max-width:660px; color:#abb7d0; font-size:1.06rem; line-height:1.65; margin-bottom:1.7rem;}
    .section-title {font-size:1.25rem; font-weight:750; color:#f1f3ff; margin:1.4rem 0 .8rem;}
    .status {display:inline-flex;align-items:center;gap:.55rem;border:1px solid #354262;border-radius:100px;padding:.45rem .8rem;background:#18213a;color:#dce5ff;font-size:.85rem;font-weight:650;}
    .dot {width:.55rem;height:.55rem;border-radius:50%;background:#efae68;}
    .dot.ready {background:#5bd8a0;}
    .dot.error {background:#fa8294;}
    .result-card {border:1px solid #39456d;border-radius:20px;background:linear-gradient(135deg,#1d2542,#151b31);padding:1.5rem;margin:.5rem 0 1.2rem;box-shadow:0 15px 35px #0002;}
    .result-label {font-size:.8rem;letter-spacing:.13em;text-transform:uppercase;color:#aeb9d5;font-weight:700;}
    .result-name {font-size:clamp(1.5rem,4vw,2.4rem);font-weight:800;color:#fff;margin:.4rem 0;word-break:break-word;}
    .result-meta {font-size:.92rem;color:#b9c3dc;}
    div.stButton > button[kind="primary"] {border-radius:12px;min-height:3rem;font-weight:750;background:linear-gradient(95deg,#7364e9,#a16df2);border:0;}
    [data-testid="stFileUploader"] {border:1px dashed #56668d;border-radius:16px;background:#151d33;padding:.5rem;}
    [data-testid="stImage"] img {border-radius:16px;max-height:360px;object-fit:contain;background:#11182a;}
    @media(max-width:640px){.block-container{padding:5rem 1rem 3rem}.result-card{padding:1.1rem}}
    </style>""", unsafe_allow_html=True)


def header():
    st.markdown('<div class="eyebrow">Artificial Intelligence · CNN</div><h1 class="hero-title">AI Image Classification</h1><p class="hero-copy">Unggah gambar, lalu gunakan model Convolutional Neural Network (CNN) kelompok untuk melihat kelas prediksi dan distribusi probabilitasnya.</p>', unsafe_allow_html=True)


def status_badge(label, kind=""):
    st.markdown(f'<span class="status"><span class="dot {kind}"></span>{label}</span>', unsafe_allow_html=True)


def result_card(result):
    from html import escape

    st.markdown(f'<div class="result-card"><div class="result-label">Predicted class</div><div class="result-name">{escape(result.label)}</div><div class="result-meta">Confidence {result.confidence:.2%} · Inference {result.inference_ms:.1f} ms</div></div>', unsafe_allow_html=True)
    st.progress(result.confidence, text=f"Confidence: {result.confidence:.2%}")
    st.caption("Confidence adalah tingkat keyakinan dari output model, bukan jaminan bahwa prediksi benar.")
    st.markdown('<div class="section-title">Probabilitas setiap kelas</div>', unsafe_allow_html=True)
    chart_data = pd.DataFrame({"Kelas": list(result.probabilities), "Probabilitas": list(result.probabilities.values())})
    st.bar_chart(chart_data, x="Kelas", y="Probabilitas", horizontal=True, x_label="Kelas", y_label="Probabilitas", color="#8b7cf6")
    for name, probability in result.probabilities.items():
        st.write(f"{name}: {probability:.2%}")
