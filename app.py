"""
Aplikasi Streamlit: Klasifikasi Skoliosis pada Citra X-Ray Tulang Belakang
Menggunakan Transfer Learning dengan Arsitektur DenseNet121

UI/UX Redesign v2:
- Layout 3 kolom proporsional (1.1 : 1.2 : 1.7)
- Satu halaman penuh TANPA SCROLL
- Kolom 1: upload + preview X-ray + info model (compact, tanpa expander)
- Kolom 2: diagnosis badge + 3 metric cards + interpretasi
- Kolom 3: heatmap & overlay Grad-CAM side-by-side
- Semua elemen fit dalam 1 viewport dengan height terkontrol
"""

import os
import io
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

os.environ["KERAS_BACKEND"] = "tensorflow"

# ─────────────────────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────────────────────
IMG_SIZE          = (224, 224)
THRESHOLD         = 0.50
POSITIVE_CLASS    = "Scoliosis"
NEGATIVE_CLASS    = "Normal"
LAST_CONV_LAYER   = "conv5_block16_concat"

GDRIVE_ID         = "1eSAJ8lDoXsm5E3K7BZlFDJq7-NloTIH6"
LOCAL_MODEL_PATH  = "models/best_densenet121_e4.keras"
CACHED_MODEL_PATH = "/tmp/best_densenet121_e4.keras"

# ─────────────────────────────────────────────────────────────
# DOWNLOAD — Google Drive
# ─────────────────────────────────────────────────────────────

def _resolve_model_path() -> str | None:
    if os.path.exists(LOCAL_MODEL_PATH):
        return LOCAL_MODEL_PATH
    if os.path.exists(CACHED_MODEL_PATH) and os.path.getsize(CACHED_MODEL_PATH) > 10_000:
        return CACHED_MODEL_PATH
    if os.path.exists(CACHED_MODEL_PATH):
        os.remove(CACHED_MODEL_PATH)
    with st.spinner("⏬ Mengunduh model dari Google Drive…"):
        try:
            import gdown
            gdown.download(
                f"https://drive.google.com/uc?id={GDRIVE_ID}",
                CACHED_MODEL_PATH,
                quiet=False,
            )
        except Exception as e:
            st.error(f"❌ Gagal mengunduh model: {e}")
            return None
    if os.path.exists(CACHED_MODEL_PATH) and os.path.getsize(CACHED_MODEL_PATH) > 10_000:
        return CACHED_MODEL_PATH
    st.error("❌ File model tidak valid setelah diunduh.")
    return None

# ─────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model():
    model_path = _resolve_model_path()
    if model_path is None:
        return None
    try:
        import keras
        return keras.models.load_model(str(model_path))
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        if os.path.exists(CACHED_MODEL_PATH):
            os.remove(CACHED_MODEL_PATH)
        return None

# ─────────────────────────────────────────────────────────────
# PRA-PROSES & PREDIKSI
# ─────────────────────────────────────────────────────────────

def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    img = pil_image.convert("RGB").resize(IMG_SIZE)
    return np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)


def predict(model, img_tensor: np.ndarray) -> tuple[str, float, float]:
    prob = float(model.predict(img_tensor, verbose=0)[0][0])
    if prob >= THRESHOLD:
        return POSITIVE_CLASS, prob * 100, prob
    return NEGATIVE_CLASS, (1 - prob) * 100, prob

# ─────────────────────────────────────────────────────────────
# GRAD-CAM
# ─────────────────────────────────────────────────────────────

def make_gradcam_heatmap(img_tensor: np.ndarray, model, prob: float) -> np.ndarray | None:
    try:
        import tensorflow as tf
        import keras
        grad_model = keras.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(LAST_CONV_LAYER).output, model.output],
        )
        img_tf = tf.cast(img_tensor, tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(img_tf)
            conv_output, predictions = grad_model(img_tf, training=False)
            p = predictions[:, 0]
            score = p if prob >= THRESHOLD else (1.0 - p)
        grads        = tape.gradient(score, conv_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap      = tf.reduce_sum(conv_output[0] * pooled_grads, axis=-1)
        heatmap      = tf.maximum(heatmap, 0).numpy()
        if heatmap.max() > 0:
            heatmap /= heatmap.max()
        return heatmap
    except Exception as e:
        st.warning(f"⚠️ Grad-CAM tidak tersedia: {e}")
        return None


def make_overlay(pil_image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    colors      = plt.colormaps["jet"](np.arange(256))[:, :3]
    jet_heatmap = Image.fromarray(np.uint8(colors[np.uint8(255 * heatmap)] * 255))
    jet_heatmap = jet_heatmap.resize(pil_image.size, Image.BILINEAR)
    return Image.blend(pil_image.convert("RGB"), jet_heatmap, alpha)


def heatmap_to_png_bytes(heatmap: np.ndarray) -> bytes:
    fig, ax = plt.subplots(figsize=(3.2, 3.2), dpi=110)
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    im = ax.imshow(heatmap, cmap="jet")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def img_to_bytes(pil_image: Image.Image) -> bytes:
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return buf.getvalue()

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Scoliosis X-Ray Classifier",
        page_icon="🩻",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # ── Global CSS ────────────────────────────────────────────
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

            html, body, [class*="css"] {
                font-family: 'Inter', sans-serif !important;
            }

            /* ── Hapus semua padding bawaan Streamlit ── */
            .block-container {
                padding: 0 !important;
                max-width: 100% !important;
            }
            [data-testid="stAppViewContainer"] {
                background: #f0f4f8 !important;
                overflow: hidden !important;
            }
            [data-testid="stHeader"],
            footer, #MainMenu,
            [data-testid="collapsedControl"] {
                display: none !important;
            }

            /* Cegah scroll pada level root */
            html, body { overflow: hidden !important; height: 100vh !important; }

            /* ── Header ── */
            .app-header {
                background: #1565c0;
                color: white;
                padding: 9px 20px;
                display: flex;
                align-items: center;
                gap: 12px;
            }
            .app-header-icon {
                width: 34px; height: 34px;
                background: rgba(255,255,255,0.15);
                border-radius: 8px;
                display: flex; align-items: center; justify-content: center;
                font-size: 18px; flex-shrink: 0;
            }
            .app-header h1 {
                margin: 0; font-size: 13px; font-weight: 700; line-height: 1.3;
            }
            .app-header p {
                margin: 0; font-size: 10px; opacity: 0.7; margin-top: 1px;
            }
            .app-header-badge {
                margin-left: auto;
                background: rgba(255,255,255,0.15);
                border: 1px solid rgba(255,255,255,0.25);
                border-radius: 20px;
                padding: 3px 10px;
                font-size: 10px; color: rgba(255,255,255,0.85);
                white-space: nowrap;
            }

            /* ── Wrapper kolom ── */
            .col-wrap {
                background: white;
                border-radius: 12px;
                border: 0.5px solid #e2e8f0;
                overflow: hidden;
                display: flex;
                flex-direction: column;
                height: calc(100vh - 70px);  /* header ~52px + padding 18px */
            }

            /* ── Panel header ── */
            .panel-hd {
                display: flex; align-items: center; gap: 7px;
                padding: 8px 12px;
                border-bottom: 0.5px solid #f0f4f8;
                flex-shrink: 0;
            }
            .panel-hd-icon {
                width: 22px; height: 22px; border-radius: 6px;
                display: flex; align-items: center; justify-content: center;
                font-size: 12px;
            }
            .panel-hd-label {
                font-size: 9.5px; font-weight: 700;
                text-transform: uppercase; letter-spacing: 1.2px; color: #94a3b8;
            }

            /* ── Upload zone ── */
            .upload-zone {
                margin: 10px;
                border: 1.5px dashed #90caf9;
                border-radius: 10px;
                background: #f0f7ff;
                padding: 24px 10px;
                text-align: center;
                cursor: pointer;
                flex-shrink: 0;
            }

            /* Override Streamlit file uploader agar compact */
            [data-testid="stFileUploader"] > div {
                border: 1.5px dashed #90caf9 !important;
                border-radius: 10px !important;
                background: #f0f7ff !important;
                padding: 6px 8px !important;
            }
            [data-testid="stFileUploader"] label { display: none !important; }
            [data-testid="stFileUploader"] small { font-size: 9px !important; }

            /* ── X-ray preview ── */
            .xray-wrap {
                margin: 8px 10px 0;
                border-radius: 10px;
                overflow: hidden;
                background: #0d1117;
                flex-shrink: 0;
            }
            .xray-wrap img {
                width: 100% !important;
                max-height: 210px !important;
                object-fit: contain !important;
                display: block;
            }

            /* ── Chip metadata ── */
            .chip {
                display: inline-block;
                background: #eff6ff; color: #1d4ed8;
                border-radius: 20px; padding: 2px 8px;
                font-size: 9.5px; font-weight: 600;
                margin-right: 4px;
            }

            /* ── Model info box (compact, selalu tampil) ── */
            .model-info {
                margin: 8px 10px;
                background: #f8fafc;
                border-radius: 8px;
                border: 0.5px solid #e2e8f0;
                padding: 8px 10px;
                flex-shrink: 0;
            }
            .model-info-title {
                font-size: 8.5px; font-weight: 700;
                text-transform: uppercase; letter-spacing: 1px;
                color: #94a3b8; margin-bottom: 5px;
            }
            .model-info-row {
                font-size: 10px; color: #64748b; margin-bottom: 2px;
                display: flex; align-items: center; gap: 5px;
            }
            .model-label {
                display: inline-flex; align-items: center; gap: 5px;
                padding: 2px 8px; border-radius: 4px;
                font-size: 10px; font-weight: 600;
            }

            /* ── Tombol utama ── */
            div[data-testid="stButton"] > button {
                width: 100%;
                font-weight: 700; font-size: 11px;
                padding: 6px 0; border-radius: 7px; border: none;
                background: #1565c0;
                color: white; margin-top: 4px;
                transition: background .15s;
                cursor: pointer;
            }
            div[data-testid="stButton"] > button:hover {
                background: #0d47a1;
            }

            /* ── Diagnosis card ── */
            .dx-card {
                margin: 8px 10px 6px;
                border-radius: 10px;
                padding: 10px 12px;
                flex-shrink: 0;
            }
            .dx-label {
                font-size: 8.5px; font-weight: 700;
                text-transform: uppercase; letter-spacing: 1px;
                color: #94a3b8; margin-bottom: 5px;
            }
            .dx-row {
                display: flex; align-items: center;
                justify-content: space-between; margin-bottom: 7px;
            }
            .dx-name {
                font-size: 18px; font-weight: 700;
                display: flex; align-items: center; gap: 6px;
            }
            .dx-pct {
                border-radius: 20px; padding: 4px 14px;
                font-size: 14px; font-weight: 700; color: white;
            }
            .conf-lbl {
                font-size: 9.5px; color: #64748b; font-weight: 600; margin-bottom: 4px;
            }
            .conf-track {
                background: rgba(0,0,0,0.08); border-radius: 6px; height: 6px;
            }
            .conf-fill {
                height: 6px; border-radius: 6px;
            }

            /* ── Metric cards ── */
            .metric-row {
                display: flex; gap: 6px;
                margin: 0 10px 6px;
                flex-shrink: 0;
            }
            .metric-card {
                flex: 1; background: #f8fafc;
                border: 0.5px solid #e2e8f0;
                border-radius: 8px; padding: 7px 6px;
                text-align: center;
            }
            .mc-l { font-size: 8px; font-weight: 700; text-transform: uppercase; letter-spacing: .8px; color: #94a3b8; }
            .mc-v { font-size: 14px; font-weight: 700; margin: 2px 0 1px; }
            .mc-s { font-size: 8px; color: #94a3b8; }

            /* ── Interpretasi box ── */
            .interp-box {
                margin: 0 10px 8px;
                background: #f8fafc;
                border-radius: 8px;
                border: 0.5px solid #e2e8f0;
                padding: 8px 10px;
                flex-shrink: 0;
            }
            .interp-title {
                font-size: 8.5px; font-weight: 700;
                text-transform: uppercase; letter-spacing: 1px;
                color: #94a3b8; margin-bottom: 4px;
            }
            .interp-text { font-size: 11px; color: #334155; line-height: 1.6; }

            /* ── Empty placeholder ── */
            .empty-ph {
                display: flex; flex-direction: column;
                align-items: center; justify-content: center;
                gap: 8px; flex: 1;
                color: #94a3b8; text-align: center;
                padding: 20px;
            }
            .empty-ph-icon { font-size: 26px; opacity: 0.3; }
            .empty-ph-text { font-size: 11px; line-height: 1.5; }

            /* ── Grad-CAM images ── */
            .gradcam-img img {
                width: 100% !important;
                max-height: 260px !important;
                object-fit: contain !important;
                border-radius: 8px;
                display: block;
            }
            .cam-sublabel {
                font-size: 8.5px; font-weight: 700;
                text-transform: uppercase; letter-spacing: 1px;
                color: #94a3b8; text-align: center; margin-bottom: 4px;
            }

            /* ── Legenda Grad-CAM ── */
            .legend-row {
                display: flex; gap: 14px; align-items: center;
                padding: 7px 10px;
                border-top: 0.5px solid #f0f4f8;
                flex-shrink: 0;
            }
            .legend-dot {
                width: 8px; height: 8px; border-radius: 50%;
                display: inline-block; margin-right: 4px;
            }
            .legend-text { font-size: 10px; color: #64748b; display: flex; align-items: center; }

            /* Streamlit kolom — hilangkan gap default */
            [data-testid="stHorizontalBlock"] > div {
                padding: 0 5px !important;
            }
            [data-testid="stHorizontalBlock"] > div:first-child { padding-left: 10px !important; }
            [data-testid="stHorizontalBlock"] > div:last-child  { padding-right: 10px !important; }

            /* Paksa kolom setinggi viewport tersedia */
            section[data-testid="stMain"] > div {
                height: calc(100vh - 52px);
                overflow: hidden;
            }
        </style>
    """, unsafe_allow_html=True)

    # ── Header ────────────────────────────────────────────────
    st.markdown("""
        <div class="app-header">
            <div class="app-header-icon">🩻</div>
            <div>
                <h1>Klasifikasi Skoliosis — Citra X-Ray Tulang Belakang</h1>
                <p>Transfer Learning · DenseNet121 · Grad-CAM Visualization · Binary Classification</p>
            </div>
            <div class="app-header-badge">DenseNet121 · TF-CPU</div>
        </div>
    """, unsafe_allow_html=True)

    # ── Layout 3 kolom ────────────────────────────────────────
    col1, col2, col3 = st.columns([1.1, 1.2, 1.7], gap="small")

    # ════════════════════════════════════════════════════════════
    # KOLOM 1 — Upload & Preview
    # ════════════════════════════════════════════════════════════
    with col1:
        st.markdown("""
            <div class="panel-hd">
                <div class="panel-hd-icon" style="background:#eff6ff;">🩻</div>
                <span class="panel-hd-label">Citra X-Ray</span>
            </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "upload", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
        )

        if uploaded_file:
            pil_image = Image.open(uploaded_file)

            # Preview X-ray
            st.markdown('<div class="xray-wrap">', unsafe_allow_html=True)
            st.image(pil_image, use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Metadata chip + tombol klasifikasi
            st.markdown(f"""
                <div style="padding:6px 10px 4px; display:flex; align-items:center; gap:4px; flex-wrap:wrap;">
                    <span class="chip">📐 {pil_image.width}×{pil_image.height}</span>
                    <span class="chip">🎨 {pil_image.mode}</span>
                </div>
            """, unsafe_allow_html=True)
            run = st.button("🔍 Klasifikasi", key="btn_classify")
        else:
            pil_image = None
            run       = False
            st.markdown("""
                <div style="margin:10px; border:1.5px dashed #90caf9; border-radius:10px;
                            padding:30px 10px; text-align:center; background:#f0f7ff;">
                    <div style="font-size:26px; opacity:.5;">🩻</div>
                    <p style="margin:6px 0 0; font-size:11px; color:#64748b; font-weight:600;">
                        Upload X-Ray Tulang Belakang
                    </p>
                    <p style="margin:2px 0 0; font-size:10px; color:#94a3b8;">JPG / PNG</p>
                </div>
            """, unsafe_allow_html=True)

        # Info model — selalu tampil, tanpa expander
        st.markdown("""
            <div class="model-info">
                <div class="model-info-title">⚙️ Info Model</div>
                <div class="model-info-row">🏗 <strong>DenseNet121</strong> (Keras + TF-CPU)</div>
                <div class="model-info-row">📄 best_densenet121_e4.keras</div>
                <div class="model-info-row">📐 Input: 224×224 px &nbsp;·&nbsp; 🎯 Thr: 0.50</div>
                <div style="display:flex; gap:5px; margin-top:6px;">
                    <span class="model-label" style="background:#e8f5e9; color:#2e7d32; flex:1; justify-content:center;">
                        🟢 Normal
                    </span>
                    <span class="model-label" style="background:#fce4e4; color:#c62828; flex:1; justify-content:center;">
                        🔴 Scoliosis
                    </span>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════
    # KOLOM 2 — Hasil Prediksi
    # ════════════════════════════════════════════════════════════
    with col2:
        st.markdown("""
            <div class="panel-hd">
                <div class="panel-hd-icon" style="background:#fce4e4;">📊</div>
                <span class="panel-hd-label">Hasil Prediksi</span>
            </div>
        """, unsafe_allow_html=True)

        # Jalankan klasifikasi saat tombol ditekan
        if uploaded_file and run:
            with st.spinner("Memuat model…"):
                model = load_model()
            if model:
                with st.spinner("Menganalisis…"):
                    img_tensor              = preprocess_image(pil_image)
                    label, confidence, prob = predict(model, img_tensor)
                st.session_state.update({
                    "label": label, "confidence": confidence, "prob": prob,
                    "img_tensor": img_tensor, "pil_image": pil_image,
                    "model": model, "ran": True, "gradcam_done": False,
                })

        if st.session_state.get("ran"):
            label      = st.session_state["label"]
            confidence = st.session_state["confidence"]
            prob       = st.session_state["prob"]

            is_sc  = label == POSITIVE_CLASS
            color  = "#c62828" if is_sc else "#2e7d32"
            bg     = "#fce4e4" if is_sc else "#e8f5e9"
            border = "#ef9a9a" if is_sc else "#a5d6a7"
            icon   = "🔴" if is_sc else "🟢"
            interp = (
                "Model mendeteksi pola kelengkungan lateral. "
                "Disarankan konfirmasi dengan evaluasi klinis lebih lanjut."
                if is_sc else
                "Tidak ditemukan indikasi kelengkungan lateral yang signifikan "
                "pada citra X-Ray ini."
            )

            # Diagnosis card
            st.markdown(f"""
                <div class="dx-card" style="background:{bg}; border:1px solid {border};">
                    <div class="dx-label">Diagnosis</div>
                    <div class="dx-row">
                        <span class="dx-name" style="color:{color};">
                            {icon} {label}
                        </span>
                        <span class="dx-pct" style="background:{color};">{confidence:.1f}%</span>
                    </div>
                    <div class="conf-lbl">Tingkat Kepercayaan</div>
                    <div class="conf-track">
                        <div class="conf-fill" style="width:{confidence:.1f}%; background:{color};"></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # 3 metric cards
            st.markdown(f"""
                <div class="metric-row">
                    <div class="metric-card">
                        <div class="mc-l">Prob. Scoliosis</div>
                        <div class="mc-v" style="color:#c62828;">{prob:.4f}</div>
                        <div class="mc-s">raw score</div>
                    </div>
                    <div class="metric-card">
                        <div class="mc-l">Prob. Normal</div>
                        <div class="mc-v" style="color:#2e7d32;">{1 - prob:.4f}</div>
                        <div class="mc-s">raw score</div>
                    </div>
                    <div class="metric-card">
                        <div class="mc-l">Threshold</div>
                        <div class="mc-v" style="color:#1565c0;">{THRESHOLD}</div>
                        <div class="mc-s">cut-off</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Interpretasi
            st.markdown(f"""
                <div class="interp-box">
                    <div class="interp-title">Interpretasi</div>
                    <div class="interp-text">{interp}</div>
                </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown("""
                <div class="empty-ph">
                    <div class="empty-ph-icon">📊</div>
                    <div class="empty-ph-text">Upload gambar lalu tekan<br><strong style="color:#475569;">Klasifikasi</strong></div>
                </div>
            """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════
    # KOLOM 3 — Grad-CAM
    # ════════════════════════════════════════════════════════════
    with col3:
        st.markdown("""
            <div class="panel-hd">
                <div class="panel-hd-icon" style="background:#fff8e1;">🔥</div>
                <span class="panel-hd-label">Grad-CAM Visualization</span>
            </div>
        """, unsafe_allow_html=True)

        if st.session_state.get("ran") and uploaded_file:

            if not st.session_state.get("gradcam_done"):
                # Placeholder + tombol generate
                st.markdown("""
                    <div class="empty-ph">
                        <div class="empty-ph-icon">🔥</div>
                        <div class="empty-ph-text">
                            Tekan tombol di bawah untuk melihat<br>area fokus model pada citra
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                if st.button("🔥 Generate Grad-CAM", key="btn_gradcam"):
                    with st.spinner("Menghasilkan peta aktivasi…"):
                        heatmap = make_gradcam_heatmap(
                            st.session_state["img_tensor"],
                            st.session_state["model"],
                            st.session_state["prob"],
                        )
                    if heatmap is not None:
                        overlay = make_overlay(st.session_state["pil_image"], heatmap)
                        st.session_state["heatmap_bytes"] = heatmap_to_png_bytes(heatmap)
                        st.session_state["overlay_bytes"] = img_to_bytes(overlay)
                        st.session_state["gradcam_done"]  = True
                        st.rerun()

            else:
                # Heatmap & Overlay side-by-side
                gc1, gc2 = st.columns(2, gap="small")

                with gc1:
                    st.markdown('<div class="cam-sublabel">🌡 Heatmap</div>', unsafe_allow_html=True)
                    st.markdown('<div class="gradcam-img">', unsafe_allow_html=True)
                    st.image(st.session_state["heatmap_bytes"], use_column_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                with gc2:
                    st.markdown('<div class="cam-sublabel">🖼 Overlay</div>', unsafe_allow_html=True)
                    st.markdown('<div class="gradcam-img">', unsafe_allow_html=True)
                    st.image(st.session_state["overlay_bytes"], use_column_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                # Legenda
                st.markdown("""
                    <div class="legend-row">
                        <span class="legend-text">
                            <span class="legend-dot" style="background:#e53e3e;"></span>
                            Merah/Kuning = Aktivasi tinggi
                        </span>
                        <span class="legend-text">
                            <span class="legend-dot" style="background:#3182ce;"></span>
                            Biru = Aktivasi rendah
                        </span>
                    </div>
                """, unsafe_allow_html=True)

                if st.button("↺ Reset Grad-CAM", key="btn_reset"):
                    st.session_state["gradcam_done"] = False
                    st.rerun()

        else:
            st.markdown("""
                <div class="empty-ph">
                    <div class="empty-ph-icon">🔥</div>
                    <div class="empty-ph-text">Selesaikan klasifikasi<br>terlebih dahulu</div>
                </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
