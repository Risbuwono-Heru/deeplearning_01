"""
Aplikasi Streamlit: Klasifikasi Skoliosis pada Citra X-Ray Tulang Belakang
Menggunakan Transfer Learning dengan Arsitektur DenseNet121

UI/UX Redesign v3:
- Satu halaman penuh, TANPA SCROLL, responsif terhadap zoom browser
- Semua ukuran relatif terhadap viewport (vh/vw) bukan pixel tetap
- Gambar X-ray dibatasi tinggi relatif agar tidak memanjang
- Layout tetap stabil saat zoom in/out
"""

import os
import io
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

os.environ["KERAS_BACKEND"] = "tensorflow"

IMG_SIZE          = (224, 224)
THRESHOLD         = 0.50
POSITIVE_CLASS    = "Scoliosis"
NEGATIVE_CLASS    = "Normal"
LAST_CONV_LAYER   = "conv5_block16_concat"

GDRIVE_ID         = "1eSAJ8lDoXsm5E3K7BZlFDJq7-NloTIH6"
LOCAL_MODEL_PATH  = "models/best_densenet121_e4.keras"
CACHED_MODEL_PATH = "/tmp/best_densenet121_e4.keras"


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


def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    img = pil_image.convert("RGB").resize(IMG_SIZE)
    return np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)


def predict(model, img_tensor: np.ndarray) -> tuple[str, float, float]:
    prob = float(model.predict(img_tensor, verbose=0)[0][0])
    if prob >= THRESHOLD:
        return POSITIVE_CLASS, prob * 100, prob
    return NEGATIVE_CLASS, (1 - prob) * 100, prob


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


def main():
    st.set_page_config(
        page_title="Scoliosis X-Ray Classifier",
        page_icon="🩻",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        /* ── Reset & prevent ALL scroll ── */
        *, *::before, *::after { box-sizing: border-box; }
        html { overflow: hidden !important; }
        body { overflow: hidden !important; font-family: 'Inter', sans-serif !important; }

        /* Hide Streamlit chrome */
        [data-testid="stHeader"], footer, #MainMenu,
        [data-testid="collapsedControl"],
        [data-testid="stToolbar"] { display: none !important; }

        /* ── App root: full viewport, no scroll ── */
        [data-testid="stAppViewContainer"] {
            background: #eef2f7 !important;
            height: 100vh !important;
            overflow: hidden !important;
        }
        [data-testid="stMain"] {
            height: 100vh !important;
            overflow: hidden !important;
        }
        section[data-testid="stMain"] > div:first-child {
            height: 100vh !important;
            overflow: hidden !important;
            padding: 0 !important;
        }

        /* ── Block container ── */
        .block-container {
            padding: 0 !important;
            max-width: 100% !important;
            height: 100vh !important;
            overflow: hidden !important;
        }

        /* ── Column wrappers: prevent overflow ── */
        [data-testid="stHorizontalBlock"] {
            height: calc(100vh - 52px) !important;
            overflow: hidden !important;
            gap: 0 !important;
            padding: 8px !important;
            align-items: stretch !important;
        }
        [data-testid="stHorizontalBlock"] > div {
            overflow: hidden !important;
            height: 100% !important;
            padding: 0 4px !important;
        }
        /* Streamlit column inner wrapper */
        [data-testid="stVerticalBlock"] {
            height: 100% !important;
            overflow: hidden !important;
            gap: 0 !important;
        }
        /* Prevent any element from pushing height */
        [data-testid="stVerticalBlock"] > div {
            flex-shrink: 0;
        }

        /* ── Header ── */
        .app-header {
            height: 52px;
            background: #1565c0;
            display: flex; align-items: center; gap: 10px;
            padding: 0 16px;
            flex-shrink: 0;
        }
        .hd-icon {
            width: 30px; height: 30px; border-radius: 7px;
            background: rgba(255,255,255,0.15);
            display: flex; align-items: center; justify-content: center;
            font-size: 16px; flex-shrink: 0;
        }
        .hd-title { font-size: 13px; font-weight: 700; color: #fff; margin: 0; }
        .hd-sub { font-size: 10px; color: rgba(255,255,255,0.65); margin: 0; }
        .hd-badge {
            margin-left: auto;
            background: rgba(255,255,255,0.15);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 20px; padding: 2px 10px;
            font-size: 10px; color: rgba(255,255,255,0.8);
            white-space: nowrap;
        }

        /* ── Panel card ── */
        .panel {
            background: #fff;
            border-radius: 10px;
            border: 0.5px solid #dde3ed;
            height: 100%;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }
        .panel-hd {
            display: flex; align-items: center; gap: 7px;
            padding: 7px 11px;
            border-bottom: 0.5px solid #eef2f7;
            flex-shrink: 0;
        }
        .panel-hd-icon {
            width: 20px; height: 20px; border-radius: 5px;
            display: flex; align-items: center; justify-content: center;
            font-size: 11px; flex-shrink: 0;
        }
        .panel-hd-label {
            font-size: 9px; font-weight: 700;
            text-transform: uppercase; letter-spacing: 1.2px; color: #94a3b8;
        }

        /* ── File uploader — ultra compact ── */
        [data-testid="stFileUploader"] {
            margin: 7px 8px 0 !important;
        }
        [data-testid="stFileUploader"] > div {
            border: 1.5px dashed #90caf9 !important;
            border-radius: 8px !important;
            background: #f0f7ff !important;
            padding: 4px 8px !important;
            min-height: 0 !important;
        }
        [data-testid="stFileUploader"] label { display: none !important; }
        [data-testid="stFileUploader"] section { padding: 4px 0 !important; }
        [data-testid="stFileUploader"] small  { font-size: 9px !important; }
        /* File name row setelah upload */
        [data-testid="stFileUploaderFileName"] { font-size: 10px !important; }

        /* ── X-ray preview: KUNCI UTAMA — tinggi relatif vh ── */
        .xray-wrap {
            margin: 6px 8px 0;
            border-radius: 8px;
            overflow: hidden;
            background: #0d1117;
            flex-shrink: 0;
            /* Tinggi relatif: 35% viewport height. 
               Saat zoom in/out nilai vh ikut skala sehingga tidak overflow */
            height: 35vh;
            display: flex; align-items: center; justify-content: center;
        }
        .xray-wrap img {
            max-width: 100% !important;
            max-height: 35vh !important;
            width: auto !important;
            height: auto !important;
            object-fit: contain !important;
            display: block;
        }
        /* Override Streamlit stImage */
        .xray-wrap [data-testid="stImage"] { height: 100% !important; }
        .xray-wrap [data-testid="stImage"] > img {
            max-height: 35vh !important;
            object-fit: contain !important;
        }

        /* ── Chip metadata ── */
        .chip {
            display: inline-block;
            background: #eff6ff; color: #1d4ed8;
            border-radius: 20px; padding: 1px 7px;
            font-size: 9px; font-weight: 600; margin-right: 3px;
        }

        /* ── Tombol utama (Klasifikasi / Grad-CAM / Reset) ── */
        div[data-testid="stButton"] > button {
            width: 100%; font-weight: 700; font-size: 11px;
            padding: 5px 0; border-radius: 7px; border: none;
            background: #1565c0; color: white;
            cursor: pointer; transition: background .15s;
        }
        div[data-testid="stButton"] > button:hover { background: #0d47a1; }

        /* ── Model info box ── */
        .model-info {
            margin: 6px 8px 0;
            background: #f8fafc;
            border: 0.5px solid #e2e8f0;
            border-radius: 7px;
            padding: 7px 9px;
            flex-shrink: 0;
        }
        .model-info-ttl {
            font-size: 8px; font-weight: 700;
            text-transform: uppercase; letter-spacing: 1px;
            color: #94a3b8; margin-bottom: 4px;
        }
        .model-info-row {
            font-size: 9.5px; color: #64748b; margin-bottom: 2px;
        }
        .model-label {
            display: inline-flex; align-items: center; gap: 4px;
            padding: 2px 7px; border-radius: 4px;
            font-size: 9.5px; font-weight: 600;
        }

        /* ── Diagnosis card ── */
        .dx-card {
            margin: 7px 8px 5px;
            border-radius: 9px; padding: 9px 11px;
            flex-shrink: 0;
        }
        .dx-lbl {
            font-size: 8px; font-weight: 700;
            text-transform: uppercase; letter-spacing: 1px;
            color: #94a3b8; margin-bottom: 4px;
        }
        .dx-row {
            display: flex; align-items: center;
            justify-content: space-between; margin-bottom: 6px;
        }
        .dx-name {
            font-size: 17px; font-weight: 700;
            display: flex; align-items: center; gap: 5px;
        }
        .dx-pct {
            border-radius: 20px; padding: 3px 12px;
            font-size: 13px; font-weight: 700; color: white;
        }
        .conf-lbl { font-size: 9px; color: #64748b; font-weight: 600; margin-bottom: 3px; }
        .conf-track { background: rgba(0,0,0,0.08); border-radius: 5px; height: 5px; }
        .conf-fill { height: 5px; border-radius: 5px; }

        /* ── Metric cards ── */
        .metric-row {
            display: flex; gap: 5px;
            margin: 0 8px 5px; flex-shrink: 0;
        }
        .metric-card {
            flex: 1; background: #f8fafc;
            border: 0.5px solid #e2e8f0;
            border-radius: 7px; padding: 6px 5px; text-align: center;
        }
        .mc-l { font-size: 7.5px; font-weight: 700; text-transform: uppercase; letter-spacing: .7px; color: #94a3b8; }
        .mc-v { font-size: 13px; font-weight: 700; margin: 2px 0 1px; }
        .mc-s { font-size: 7.5px; color: #94a3b8; }

        /* ── Interpretasi ── */
        .interp-box {
            margin: 0 8px 6px;
            background: #f8fafc; border: 0.5px solid #e2e8f0;
            border-radius: 7px; padding: 7px 9px; flex-shrink: 0;
        }
        .interp-ttl {
            font-size: 8px; font-weight: 700;
            text-transform: uppercase; letter-spacing: 1px;
            color: #94a3b8; margin-bottom: 3px;
        }
        .interp-txt { font-size: 10.5px; color: #334155; line-height: 1.55; }

        /* ── Empty placeholder ── */
        .empty-ph {
            display: flex; flex-direction: column;
            align-items: center; justify-content: center;
            gap: 7px; padding: 16px; color: #94a3b8;
            text-align: center; flex: 1;
        }
        .empty-ph-icon { font-size: 24px; opacity: 0.28; }
        .empty-ph-txt { font-size: 10.5px; line-height: 1.5; }

        /* ── Grad-CAM gambar ── */
        .cam-sub {
            font-size: 8px; font-weight: 700;
            text-transform: uppercase; letter-spacing: 1px;
            color: #94a3b8; text-align: center;
            margin: 6px 0 3px;
        }
        .gradcam-img {
            border-radius: 7px; overflow: hidden;
            background: #0d1117;
            /* Tinggi relatif vh agar ikut zoom */
            height: 32vh;
            display: flex; align-items: center; justify-content: center;
        }
        .gradcam-img img,
        .gradcam-img [data-testid="stImage"] > img {
            max-width: 100% !important;
            max-height: 32vh !important;
            width: auto !important;
            height: auto !important;
            object-fit: contain !important;
            display: block;
        }

        /* ── Legenda ── */
        .legend-row {
            display: flex; gap: 12px; align-items: center;
            padding: 5px 8px; border-top: 0.5px solid #eef2f7;
            flex-shrink: 0;
        }
        .legend-dot {
            width: 7px; height: 7px; border-radius: 50%;
            display: inline-block; margin-right: 3px;
        }
        .legend-txt { font-size: 9.5px; color: #64748b; display: flex; align-items: center; }

        /* Paksa st.image tidak punya margin extra */
        [data-testid="stImage"] { margin: 0 !important; line-height: 0 !important; }
    </style>
    """, unsafe_allow_html=True)

    # ── Header ────────────────────────────────────────────────
    st.markdown("""
    <div class="app-header">
        <div class="hd-icon">🩻</div>
        <div>
            <p class="hd-title">Klasifikasi Skoliosis — Citra X-Ray Tulang Belakang</p>
            <p class="hd-sub">Transfer Learning · DenseNet121 · Grad-CAM Visualization · Binary Classification</p>
        </div>
        <div class="hd-badge">DenseNet121 · TF-CPU</div>
    </div>
    """, unsafe_allow_html=True)

    # ── 3 kolom ───────────────────────────────────────────────
    col1, col2, col3 = st.columns([1.1, 1.2, 1.7], gap="small")

    # ════════════════════════════════════════════════════
    # KOLOM 1 — Upload & Preview
    # ════════════════════════════════════════════════════
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

            # Wrap gambar dalam div ber-height relatif vh
            st.markdown('<div class="xray-wrap">', unsafe_allow_html=True)
            st.image(pil_image, use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Metadata + tombol
            st.markdown(f"""
            <div style="padding:4px 8px 3px; display:flex; align-items:center; gap:3px; flex-wrap:wrap;">
                <span class="chip">📐 {pil_image.width}×{pil_image.height}</span>
                <span class="chip">🎨 {pil_image.mode}</span>
            </div>
            """, unsafe_allow_html=True)

            run = st.button("🔍 Klasifikasi", key="btn_classify")
        else:
            pil_image = None
            run       = False
            st.markdown("""
            <div style="margin:7px 8px 0; border:1.5px dashed #90caf9; border-radius:8px;
                        padding:22px 10px; text-align:center; background:#f0f7ff;">
                <div style="font-size:22px; opacity:.45;">🩻</div>
                <p style="margin:5px 0 0; font-size:10.5px; color:#64748b; font-weight:600;">
                    Upload X-Ray Tulang Belakang
                </p>
                <p style="margin:2px 0 0; font-size:9.5px; color:#94a3b8;">JPG / PNG</p>
            </div>
            """, unsafe_allow_html=True)

        # Info model — selalu tampil
        st.markdown("""
        <div class="model-info">
            <div class="model-info-ttl">⚙️ Info Model</div>
            <div class="model-info-row">🏗 <b>DenseNet121</b> (Keras + TF-CPU)</div>
            <div class="model-info-row">📄 best_densenet121_e4.keras</div>
            <div class="model-info-row">📐 224×224 px &nbsp;·&nbsp; 🎯 Thr: 0.50</div>
            <div style="display:flex; gap:5px; margin-top:5px;">
                <span class="model-label" style="background:#e8f5e9;color:#2e7d32;flex:1;justify-content:center;">
                    🟢 Normal
                </span>
                <span class="model-label" style="background:#fce4e4;color:#c62828;flex:1;justify-content:center;">
                    🔴 Scoliosis
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════
    # KOLOM 2 — Hasil Prediksi
    # ════════════════════════════════════════════════════
    with col2:
        st.markdown("""
        <div class="panel-hd">
            <div class="panel-hd-icon" style="background:#fce4e4;">📊</div>
            <span class="panel-hd-label">Hasil Prediksi</span>
        </div>
        """, unsafe_allow_html=True)

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
                "Model mendeteksi pola kelengkungan lateral pada citra. "
                "Disarankan konfirmasi dengan evaluasi klinis lebih lanjut."
                if is_sc else
                "Tidak ditemukan indikasi kelengkungan lateral yang signifikan pada citra ini."
            )

            st.markdown(f"""
            <div class="dx-card" style="background:{bg}; border:1px solid {border};">
                <div class="dx-lbl">Diagnosis</div>
                <div class="dx-row">
                    <span class="dx-name" style="color:{color};">{icon} {label}</span>
                    <span class="dx-pct" style="background:{color};">{confidence:.1f}%</span>
                </div>
                <div class="conf-lbl">Tingkat Kepercayaan</div>
                <div class="conf-track">
                    <div class="conf-fill" style="width:{confidence:.1f}%; background:{color};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="metric-row">
                <div class="metric-card">
                    <div class="mc-l">Prob. Scoliosis</div>
                    <div class="mc-v" style="color:#c62828;">{prob:.4f}</div>
                    <div class="mc-s">raw score</div>
                </div>
                <div class="metric-card">
                    <div class="mc-l">Prob. Normal</div>
                    <div class="mc-v" style="color:#2e7d32;">{1-prob:.4f}</div>
                    <div class="mc-s">raw score</div>
                </div>
                <div class="metric-card">
                    <div class="mc-l">Threshold</div>
                    <div class="mc-v" style="color:#1565c0;">{THRESHOLD}</div>
                    <div class="mc-s">cut-off</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="interp-box">
                <div class="interp-ttl">Interpretasi</div>
                <div class="interp-txt">{interp}</div>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="empty-ph">
                <div class="empty-ph-icon">📊</div>
                <div class="empty-ph-txt">
                    Upload gambar lalu tekan<br>
                    <strong style="color:#475569;">Klasifikasi</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════
    # KOLOM 3 — Grad-CAM
    # ════════════════════════════════════════════════════
    with col3:
        st.markdown("""
        <div class="panel-hd">
            <div class="panel-hd-icon" style="background:#fff8e1;">🔥</div>
            <span class="panel-hd-label">Grad-CAM Visualization</span>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.get("ran") and uploaded_file:

            if not st.session_state.get("gradcam_done"):
                st.markdown("""
                <div class="empty-ph">
                    <div class="empty-ph-icon">🔥</div>
                    <div class="empty-ph-txt">
                        Tekan tombol untuk melihat<br>area fokus model pada citra
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
                gc1, gc2 = st.columns(2, gap="small")

                with gc1:
                    st.markdown('<div class="cam-sub">🌡 Heatmap</div>', unsafe_allow_html=True)
                    st.markdown('<div class="gradcam-img">', unsafe_allow_html=True)
                    st.image(st.session_state["heatmap_bytes"], use_column_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                with gc2:
                    st.markdown('<div class="cam-sub">🖼 Overlay</div>', unsafe_allow_html=True)
                    st.markdown('<div class="gradcam-img">', unsafe_allow_html=True)
                    st.image(st.session_state["overlay_bytes"], use_column_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                st.markdown("""
                <div class="legend-row">
                    <span class="legend-txt">
                        <span class="legend-dot" style="background:#e53e3e;"></span>
                        Merah/Kuning = Aktivasi tinggi
                    </span>
                    <span class="legend-txt">
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
                <div class="empty-ph-txt">Selesaikan klasifikasi<br>terlebih dahulu</div>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
