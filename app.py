"""
Aplikasi Streamlit: Klasifikasi Skoliosis pada Citra X-Ray Tulang Belakang
Menggunakan Transfer Learning dengan Arsitektur DenseNet121

UI/UX Redesign:
- Layout 3 kolom dengan rasio proporsional (1.1 : 1.2 : 1.7)
- Kolom 1: gambar besar + tombol klasifikasi, info sekunder dalam expander
- Kolom 2: hasil prediksi mengisi penuh ruang + metric cards
- Kolom 3: Grad-CAM lebih lebar, heatmap & overlay side-by-side proporsional
- Tidak ada scroll — semua muat dalam 1 viewport
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

    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
            html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

            /* Hapus padding default Streamlit, paksa viewport-fit */
            .block-container {
                padding: 0 1rem 0 1rem !important;
                max-width: 100% !important;
            }
            [data-testid="stAppViewContainer"] { background: #f0f4f8; }
            [data-testid="stHeader"] { display: none; }
            footer, #MainMenu, [data-testid="collapsedControl"] { display: none !important; }

            /* ── Header compact ── */
            .app-header {
                background: linear-gradient(135deg, #0d47a1 0%, #1976d2 100%);
                color: white; padding: 10px 20px;
                border-radius: 0 0 10px 10px; margin-bottom: 10px;
                display: flex; align-items: center; gap: 12px;
                box-shadow: 0 3px 12px rgba(13,71,161,0.22);
            }
            .app-header h1 { margin: 0; font-size: 0.98rem; font-weight: 700; line-height: 1.3; }
            .app-header p  { margin: 0; font-size: 0.7rem; opacity: 0.78; }

            /* ── Panel card ── */
            .panel {
                background: white; border-radius: 12px;
                padding: 12px 14px; height: 100%;
                box-shadow: 0 1px 5px rgba(0,0,0,0.07);
            }
            .sec-title {
                font-size: 0.6rem; font-weight: 700; text-transform: uppercase;
                letter-spacing: 1.3px; color: #94a3b8; margin-bottom: 8px;
                display: flex; align-items: center; gap: 5px;
            }

            /* ── Upload area ── */
            [data-testid="stFileUploader"] > div {
                border: 2px dashed #90caf9 !important;
                border-radius: 10px !important;
                background: #f0f7ff !important;
                padding: 4px !important;
            }
            [data-testid="stFileUploader"] label { display: none !important; }

            /* ── Gambar X-ray — tinggi tetap, proporsional ── */
            .xray-wrap img {
                width: 100% !important;
                max-height: 340px !important;
                object-fit: contain !important;
                border-radius: 8px;
                background: #0d1117;
            }

            /* ── Gambar Grad-CAM ── */
            .gradcam-img img {
                width: 100% !important;
                max-height: 280px !important;
                object-fit: contain !important;
                border-radius: 8px;
            }

            /* ── Chip ── */
            .chip {
                display: inline-block; background: #e8f0fe; color: #1a56db;
                padding: 1px 7px; border-radius: 10px;
                font-size: 0.68rem; font-weight: 600; margin: 1px 2px 0 0;
            }

            /* ── Tombol Klasifikasi ── */
            div[data-testid="stButton"] > button {
                width: 100%; font-weight: 700; font-size: 0.88rem;
                padding: 8px 0; border-radius: 8px; border: none;
                background: linear-gradient(135deg,#1565c0,#1976d2);
                color: white; margin-top: 6px;
                box-shadow: 0 2px 8px rgba(21,101,192,0.28);
                transition: all .18s;
            }
            div[data-testid="stButton"] > button:hover {
                background: linear-gradient(135deg,#0d47a1,#1565c0);
                transform: translateY(-1px);
            }

            /* ── Metric cards (3 kotak prob) ── */
            .metric-row {
                display: flex; gap: 8px; margin-top: 10px;
            }
            .metric-card {
                flex: 1; background: #f8fafc; border: 1px solid #e2e8f0;
                border-radius: 10px; padding: 10px 10px 8px;
                text-align: center;
            }
            .metric-card .mc-label {
                font-size: 0.6rem; font-weight: 700; text-transform: uppercase;
                letter-spacing: .9px; color: #94a3b8; margin-bottom: 4px;
            }
            .metric-card .mc-value {
                font-size: 1.15rem; font-weight: 700; color: #1e293b;
            }
            .metric-card .mc-sub {
                font-size: 0.6rem; color: #94a3b8; margin-top: 1px;
            }

            /* ── Legenda Grad-CAM ── */
            .legend-row {
                display: flex; gap: 14px; align-items: center;
                font-size: 0.7rem; color: #64748b; margin-top: 8px;
            }
            .legend-dot {
                width: 9px; height: 9px; border-radius: 50%;
                display: inline-block; margin-right: 4px;
            }

            /* ── Expander info model ── */
            [data-testid="stExpander"] {
                border: 1px solid #e2e8f0 !important;
                border-radius: 8px !important;
                margin-top: 8px !important;
            }
            [data-testid="stExpander"] summary {
                font-size: 0.72rem !important; color: #475569 !important; padding: 6px 10px !important;
            }
        </style>
    """, unsafe_allow_html=True)

    # ── Header ────────────────────────────────────────────────
    st.markdown("""
        <div class="app-header">
            <span style="font-size:1.9rem;">🩻</span>
            <div>
                <h1>Klasifikasi Skoliosis — Citra X-Ray Tulang Belakang</h1>
                <p>Transfer Learning · DenseNet121 · Grad-CAM Visualization · Binary Classification</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # ── Layout: kolom 1.1 : 1.2 : 1.7 ────────────────────────
    col_upload, col_result, col_gradcam = st.columns([1.1, 1.2, 1.7], gap="medium")

    # ════════════════════════════════════════════════════════════
    # KOLOM 1 — Upload & Preview
    # ════════════════════════════════════════════════════════════
    with col_upload:
        st.markdown('<div class="sec-title">📁 Citra X-Ray</div>', unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "upload", type=["jpg","jpeg","png"], label_visibility="collapsed"
        )

        if uploaded_file:
            pil_image = Image.open(uploaded_file)

            # Gambar preview dengan max-height via wrapper class
            st.markdown('<div class="xray-wrap">', unsafe_allow_html=True)
            st.image(pil_image, use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Chip metadata + tombol klasifikasi sejajar
            c1, c2 = st.columns([3, 2])
            with c1:
                st.markdown(f"""
                    <div style="margin-top:5px;">
                        <span class="chip">📐 {pil_image.width}×{pil_image.height}</span>
                        <span class="chip">🎨 {pil_image.mode}</span>
                    </div>
                """, unsafe_allow_html=True)
            with c2:
                run = st.button("🔍 Klasifikasi")
        else:
            pil_image = None
            run = False
            st.markdown("""
                <div style="border:2px dashed #90caf9; border-radius:10px;
                            padding:52px 10px; text-align:center; background:#f0f7ff;">
                    <div style="font-size:2.2rem;">🩻</div>
                    <p style="margin:6px 0 0; font-size:0.8rem; color:#64748b; font-weight:600;">
                        Upload X-Ray Tulang Belakang
                    </p>
                    <p style="margin:2px 0 0; font-size:0.7rem; color:#94a3b8;">JPG / PNG</p>
                </div>
            """, unsafe_allow_html=True)

        # Info model dalam expander agar tidak makan ruang
        with st.expander("⚙️ Info Model"):
            st.markdown("""
                <div style="font-size:0.74rem; color:#475569; line-height:1.8;">
                    🏗 <strong>DenseNet121</strong> (Keras + TF-CPU)<br>
                    📄 best_densenet121_e4.keras<br>
                    📐 Input: 224×224 px<br>
                    🎯 Threshold: 0.50
                </div>
                <div style="margin-top:8px;">
                    <div style="background:#e8f5e9;border-left:3px solid #2e7d32;border-radius:4px;
                                padding:4px 8px;font-size:0.7rem;margin-bottom:4px;">
                        🟢 <strong>Normal</strong> — Tidak ada kelengkungan lateral
                    </div>
                    <div style="background:#fce4e4;border-left:3px solid #c62828;border-radius:4px;
                                padding:4px 8px;font-size:0.7rem;">
                        🔴 <strong>Scoliosis</strong> — Terdeteksi kelengkungan lateral
                    </div>
                </div>
            """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════
    # KOLOM 2 — Hasil Prediksi
    # ════════════════════════════════════════════════════════════
    with col_result:
        st.markdown('<div class="sec-title">📊 Hasil Prediksi</div>', unsafe_allow_html=True)

        # Jalankan klasifikasi
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

            # ── Diagnosis badge utama ──
            st.markdown(f"""
                <div style="background:{bg}; border:1.5px solid {border};
                            border-radius:12px; padding:14px 16px;
                            box-shadow:0 2px 8px rgba(0,0,0,0.05);">
                    <p style="margin:0;font-size:0.6rem;font-weight:700;text-transform:uppercase;
                               letter-spacing:1px;color:#94a3b8;">Diagnosis</p>
                    <div style="display:flex;align-items:center;justify-content:space-between;margin:5px 0 8px;">
                        <span style="font-size:1.55rem;font-weight:700;color:{color};">{icon} {label}</span>
                        <span style="background:{color};color:white;padding:5px 16px;
                                      border-radius:20px;font-size:1rem;font-weight:700;">
                            {confidence:.1f}%
                        </span>
                    </div>
                    <p style="margin:0 0 4px;font-size:0.7rem;color:#64748b;font-weight:600;">
                        Tingkat Kepercayaan
                    </p>
                    <div style="background:rgba(0,0,0,0.08);border-radius:6px;height:8px;">
                        <div style="background:{color};width:{confidence:.1f}%;
                                    height:8px;border-radius:6px;
                                    transition:width .5s ease;"></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # ── 3 metric cards ──
            st.markdown(f"""
                <div class="metric-row">
                    <div class="metric-card">
                        <div class="mc-label">Prob. Scoliosis</div>
                        <div class="mc-value" style="color:#c62828;">{prob:.4f}</div>
                        <div class="mc-sub">raw score</div>
                    </div>
                    <div class="metric-card">
                        <div class="mc-label">Prob. Normal</div>
                        <div class="mc-value" style="color:#2e7d32;">{1-prob:.4f}</div>
                        <div class="mc-sub">raw score</div>
                    </div>
                    <div class="metric-card">
                        <div class="mc-label">Threshold</div>
                        <div class="mc-value" style="color:#1565c0;">{THRESHOLD}</div>
                        <div class="mc-sub">cut-off</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # ── Interpretasi teks ──
            interp = (
                "Model mendeteksi pola kelengkungan lateral pada citra. "
                "Disarankan konfirmasi dengan evaluasi klinis lebih lanjut."
                if is_sc else
                "Tidak ditemukan indikasi kelengkungan lateral yang signifikan "
                "pada citra X-Ray ini."
            )
            st.markdown(f"""
                <div style="margin-top:10px;background:#f8fafc;border-radius:10px;
                            padding:10px 12px;border:1px solid #e2e8f0;">
                    <p style="margin:0;font-size:0.62rem;font-weight:700;text-transform:uppercase;
                               letter-spacing:.9px;color:#94a3b8;margin-bottom:4px;">Interpretasi</p>
                    <p style="margin:0;font-size:0.76rem;color:#334155;line-height:1.55;">{interp}</p>
                </div>
            """, unsafe_allow_html=True)

        else:
            # State kosong — placeholder proporsional
            st.markdown("""
                <div style="border:1px solid #e2e8f0;border-radius:12px;
                            padding:80px 10px;text-align:center;background:#fafafa;">
                    <div style="font-size:2.4rem;opacity:.25;">📊</div>
                    <p style="margin:8px 0 0;font-size:0.8rem;color:#94a3b8;">
                        Upload gambar lalu tekan <strong>Klasifikasi</strong>
                    </p>
                </div>
            """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════
    # KOLOM 3 — Grad-CAM (lebar 1.7)
    # ════════════════════════════════════════════════════════════
    with col_gradcam:
        st.markdown('<div class="sec-title">🔥 Grad-CAM Visualization</div>', unsafe_allow_html=True)

        if st.session_state.get("ran") and uploaded_file:

            if not st.session_state.get("gradcam_done"):
                # Tombol generate — tampil ketika belum ada hasil
                if st.button("🔥 Generate Grad-CAM"):
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

                # Placeholder sebelum di-generate
                st.markdown("""
                    <div style="border:1px solid #e2e8f0;border-radius:12px;
                                padding:80px 10px;text-align:center;background:#fafafa;">
                        <div style="font-size:2.4rem;opacity:.25;">🔥</div>
                        <p style="margin:8px 0 0;font-size:0.8rem;color:#94a3b8;">
                            Tekan <strong>Generate Grad-CAM</strong> untuk melihat<br>area fokus model
                        </p>
                    </div>
                """, unsafe_allow_html=True)

            else:
                # ── Grad-CAM sudah ada — tampilkan 2 gambar side by side ──
                gc1, gc2 = st.columns(2, gap="small")

                with gc1:
                    st.markdown("""
                        <p style="font-size:0.62rem;font-weight:700;text-transform:uppercase;
                                   letter-spacing:1px;color:#94a3b8;text-align:center;margin-bottom:4px;">
                            🌡 Heatmap
                        </p>
                    """, unsafe_allow_html=True)
                    st.markdown('<div class="gradcam-img">', unsafe_allow_html=True)
                    st.image(st.session_state["heatmap_bytes"], use_column_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                with gc2:
                    st.markdown("""
                        <p style="font-size:0.62rem;font-weight:700;text-transform:uppercase;
                                   letter-spacing:1px;color:#94a3b8;text-align:center;margin-bottom:4px;">
                            🖼 Overlay
                        </p>
                    """, unsafe_allow_html=True)
                    st.markdown('<div class="gradcam-img">', unsafe_allow_html=True)
                    st.image(st.session_state["overlay_bytes"], use_column_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                # Legenda + tombol reset
                st.markdown("""
                    <div class="legend-row">
                        <span><span class="legend-dot" style="background:#e53e3e;"></span>Merah/Kuning = Aktivasi tinggi</span>
                        <span><span class="legend-dot" style="background:#3182ce;"></span>Biru = Aktivasi rendah</span>
                    </div>
                """, unsafe_allow_html=True)

                # Tombol reset kecil di bawah legenda
                if st.button("↺ Reset Grad-CAM"):
                    st.session_state["gradcam_done"] = False
                    st.rerun()

        else:
            st.markdown("""
                <div style="border:1px solid #e2e8f0;border-radius:12px;
                            padding:80px 10px;text-align:center;background:#fafafa;">
                    <div style="font-size:2.4rem;opacity:.25;">🔥</div>
                    <p style="margin:8px 0 0;font-size:0.8rem;color:#94a3b8;">
                        Selesaikan klasifikasi terlebih dahulu
                    </p>
                </div>
            """, unsafe_allow_html=True)


def img_to_bytes(pil_image: Image.Image) -> bytes:
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return buf.getvalue()


if __name__ == "__main__":
    main()
