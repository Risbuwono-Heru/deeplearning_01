"""
Aplikasi Streamlit: Klasifikasi Skoliosis pada Citra X-Ray Tulang Belakang
Menggunakan Transfer Learning dengan Arsitektur DenseNet121
"""

import os
import gdown
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

os.environ["KERAS_BACKEND"] = "jax"
import keras

# ─────────────────────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────────────────────
IMG_SIZE          = (224, 224)
THRESHOLD         = 0.50
POSITIVE_CLASS    = "Scoliosis"
NEGATIVE_CLASS    = "Normal"
GDRIVE_ID         = "1eSAJ8lDoXsm5E3K7BZlFDJq7-NloTIH6"
LOCAL_MODEL_PATH  = "models/best_densenet121_e4.keras"
CACHED_MODEL_PATH = "/tmp/best_densenet121_e4.keras"
LAST_CONV_LAYER   = "conv5_block16_concat"

# ─────────────────────────────────────────────────────────────
# DOWNLOAD MODEL
# ─────────────────────────────────────────────────────────────

def resolve_model_path():
    if os.path.exists(LOCAL_MODEL_PATH):
        return LOCAL_MODEL_PATH

    if os.path.exists(CACHED_MODEL_PATH) and os.path.getsize(CACHED_MODEL_PATH) > 10_000:
        return CACHED_MODEL_PATH

    if os.path.exists(CACHED_MODEL_PATH):
        os.remove(CACHED_MODEL_PATH)

    with st.spinner("⏬ Mengunduh model dari Google Drive..."):
        try:
            url = f"https://drive.google.com/uc?id={GDRIVE_ID}"
            gdown.download(url, CACHED_MODEL_PATH, quiet=False)
        except Exception as e:
            st.error(f"Gagal mengunduh model: {e}")
            return None

    if os.path.exists(CACHED_MODEL_PATH) and os.path.getsize(CACHED_MODEL_PATH) > 10_000:
        return CACHED_MODEL_PATH

    st.error("❌ File model tidak valid setelah diunduh.")
    return None


# ─────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model():
    model_path = resolve_model_path()
    if model_path is None:
        return None
    try:
        return keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        if os.path.exists(CACHED_MODEL_PATH):
            os.remove(CACHED_MODEL_PATH)
        return None


# ─────────────────────────────────────────────────────────────
# PRA-PROSES & PREDIKSI
# ─────────────────────────────────────────────────────────────

def preprocess_image(pil_image):
    img = pil_image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def predict(model, img_tensor):
    prob = float(model.predict(img_tensor, verbose=0)[0][0])
    if prob >= THRESHOLD:
        return POSITIVE_CLASS, prob * 100, prob
    return NEGATIVE_CLASS, (1 - prob) * 100, prob


# ─────────────────────────────────────────────────────────────
# GRAD-CAM
# ─────────────────────────────────────────────────────────────

def make_gradcam_heatmap(img_tensor, model, prob):
    try:
        import jax
        import jax.numpy as jnp

        grad_model = keras.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(LAST_CONV_LAYER).output, model.output],
        )
        img_jax = jnp.array(img_tensor)

        def forward(x):
            conv_out, pred = grad_model(x, training=False)
            p = pred[:, 0]
            return (p if prob >= THRESHOLD else (1 - p)).sum(), conv_out

        (_, conv_output), grads = jax.value_and_grad(forward, has_aux=True)(img_jax)
        pooled_grads = jnp.mean(grads[1], axis=(0, 1, 2))
        heatmap      = jnp.sum(conv_output[0] * pooled_grads, axis=-1)
        heatmap      = np.array(jnp.maximum(heatmap, 0))
        if heatmap.max() > 0:
            heatmap /= heatmap.max()
        return heatmap

    except Exception as e:
        st.warning(f"Grad-CAM tidak tersedia: {e}")
        return None


def overlay_gradcam(pil_image, heatmap, alpha=0.45):
    colors      = cm.get_cmap("jet")(np.arange(256))[:, :3]
    jet_heatmap = Image.fromarray(np.uint8(colors[np.uint8(255 * heatmap)] * 255))
    jet_heatmap = jet_heatmap.resize(pil_image.size, Image.BILINEAR)
    return Image.blend(pil_image.convert("RGB"), jet_heatmap, alpha)


# ─────────────────────────────────────────────────────────────
# TAMPILAN HASIL
# ─────────────────────────────────────────────────────────────

def show_result(label, confidence, prob):
    is_scoliosis = label == POSITIVE_CLASS
    color        = "#c62828" if is_scoliosis else "#2e7d32"
    bg_color     = "#fce4e4" if is_scoliosis else "#e8f5e9"
    border_color = "#ef9a9a" if is_scoliosis else "#a5d6a7"
    icon         = "🔴" if is_scoliosis else "🟢"
    badge_bg     = "#c62828" if is_scoliosis else "#2e7d32"

    st.markdown(f"""
        <div style="background:{bg_color}; border:1.5px solid {border_color};
                    border-radius:12px; padding:22px 26px; margin-top:12px;
                    box-shadow:0 2px 8px rgba(0,0,0,0.07);">
            <div style="display:flex; align-items:center; gap:12px; margin-bottom:12px;">
                <span style="font-size:2rem;">{icon}</span>
                <div>
                    <p style="margin:0; font-size:0.75rem; color:#666; text-transform:uppercase;
                               letter-spacing:1px; font-weight:600;">Hasil Klasifikasi</p>
                    <h2 style="margin:0; color:{color}; font-size:1.6rem;">{label}</h2>
                </div>
                <span style="margin-left:auto; background:{badge_bg}; color:white;
                              padding:4px 14px; border-radius:20px; font-size:0.85rem;
                              font-weight:700;">{confidence:.1f}%</span>
            </div>
            <p style="margin:0 0 6px 0; font-size:0.82rem; color:#555; font-weight:600;">
                Tingkat Kepercayaan
            </p>
            <div style="background:rgba(0,0,0,0.08); border-radius:8px; height:12px; width:100%;">
                <div style="background:{color}; width:{confidence:.1f}%; height:12px;
                            border-radius:8px; transition:width .5s;"></div>
            </div>
            <p style="margin:10px 0 0 0; font-size:0.78rem; color:#777;">
                Probabilitas Scoliosis: <strong>{prob:.4f}</strong> &nbsp;|&nbsp;
                Probabilitas Normal: <strong>{1-prob:.4f}</strong> &nbsp;|&nbsp;
                Threshold: <strong>{THRESHOLD}</strong>
            </p>
        </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="Scoliosis X-Ray Classifier", page_icon="🩻", layout="wide")

    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

            html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

            .block-container { padding-top: 0 !important; padding-bottom: 2rem; }

            /* Header banner */
            .app-header {
                background: linear-gradient(135deg, #0d47a1 0%, #1565c0 50%, #1976d2 100%);
                color: white;
                padding: 28px 40px;
                border-radius: 0 0 16px 16px;
                margin-bottom: 24px;
                box-shadow: 0 4px 20px rgba(13,71,161,0.3);
            }
            .app-header h1 {
                margin: 0 0 6px 0;
                font-size: 1.6rem;
                font-weight: 700;
                letter-spacing: -0.3px;
            }
            .app-header p { margin: 0; font-size: 0.9rem; opacity: 0.85; }

            /* Cards */
            .info-card {
                background: white;
                border: 1px solid #e8ecf0;
                border-radius: 12px;
                padding: 16px 20px;
                margin-bottom: 12px;
                box-shadow: 0 1px 4px rgba(0,0,0,0.06);
            }

            /* Sidebar */
            section[data-testid="stSidebar"] {
                background: #f8fafc;
                border-right: 1px solid #e2e8f0;
            }
            section[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }

            /* Upload area */
            [data-testid="stFileUploader"] {
                border: 2px dashed #90caf9 !important;
                border-radius: 12px !important;
                background: #f0f7ff !important;
                padding: 12px !important;
            }

            /* Classify button */
            .stButton > button {
                width: 100%;
                background: linear-gradient(135deg, #1565c0, #1976d2);
                color: white;
                border: none;
                border-radius: 10px;
                padding: 12px 0;
                font-size: 1rem;
                font-weight: 600;
                margin-top: 10px;
                box-shadow: 0 3px 10px rgba(21,101,192,0.3);
                transition: all .2s;
            }
            .stButton > button:hover {
                background: linear-gradient(135deg, #0d47a1, #1565c0);
                box-shadow: 0 5px 15px rgba(21,101,192,0.4);
                transform: translateY(-1px);
            }

            /* Section titles */
            .section-title {
                font-size: 0.7rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 1.2px;
                color: #64748b;
                margin-bottom: 10px;
            }

            /* Metric chips */
            .metric-chip {
                display: inline-block;
                background: #e8f0fe;
                color: #1a56db;
                padding: 3px 10px;
                border-radius: 20px;
                font-size: 0.78rem;
                font-weight: 600;
                margin: 2px 2px;
            }

            /* Divider */
            hr { border: none; border-top: 1px solid #e2e8f0; margin: 14px 0; }
        </style>
    """, unsafe_allow_html=True)

    # ── Header Banner ─────────────────────────────────────────
    st.markdown("""
        <div class="app-header">
            <h1>🩻 Klasifikasi Skoliosis pada Citra X-Ray Tulang Belakang</h1>
            <p>Transfer Learning · DenseNet121 · Grad-CAM Visualization · Binary Classification</p>
        </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
            <div style="text-align:center; padding:8px 0 16px 0;">
                <span style="font-size:2.5rem;">🩻</span>
                <p style="margin:4px 0 0 0; font-weight:700; font-size:1rem; color:#1e293b;">
                    Scoliosis Classifier
                </p>
                <p style="margin:2px 0 0 0; font-size:0.75rem; color:#64748b;">
                    Deep Learning · X-Ray Analysis
                </p>
            </div>
            <hr>
        """, unsafe_allow_html=True)

        st.markdown('<p class="section-title">⚙️ Konfigurasi Model</p>', unsafe_allow_html=True)
        st.markdown("""
            <div class="info-card">
                <p style="margin:0 0 8px 0; font-size:0.82rem; color:#64748b; font-weight:600;">ARSITEKTUR</p>
                <p style="margin:0; font-weight:700; color:#1e293b; font-size:0.95rem;">DenseNet121</p>
                <hr>
                <p style="margin:0 0 4px 0; font-size:0.78rem; color:#64748b;">
                    📄 best_densenet121_e4.keras
                </p>
                <p style="margin:0 0 4px 0; font-size:0.78rem; color:#64748b;">
                    📐 Input: 224 × 224 px
                </p>
                <p style="margin:0; font-size:0.78rem; color:#64748b;">
                    🎯 Threshold: 0.50
                </p>
            </div>
        """, unsafe_allow_html=True)

        show_gradcam = st.toggle("🔥 Tampilkan Grad-CAM", value=True)

        st.markdown('<hr><p class="section-title">📋 Kelas Output</p>', unsafe_allow_html=True)

        st.markdown("""
            <div style="background:#e8f5e9; border-left:4px solid #2e7d32;
                        border-radius:8px; padding:10px 14px; margin-bottom:8px;">
                <p style="margin:0; font-weight:700; color:#2e7d32; font-size:0.9rem;">🟢 Normal</p>
                <p style="margin:3px 0 0 0; font-size:0.75rem; color:#4caf50;">
                    Tidak terdeteksi kelengkungan lateral
                </p>
            </div>
            <div style="background:#fce4e4; border-left:4px solid #c62828;
                        border-radius:8px; padding:10px 14px; margin-bottom:8px;">
                <p style="margin:0; font-weight:700; color:#c62828; font-size:0.9rem;">🔴 Scoliosis</p>
                <p style="margin:3px 0 0 0; font-size:0.75rem; color:#ef5350;">
                    Terdeteksi kelengkungan lateral pada tulang belakang
                </p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown('<hr>', unsafe_allow_html=True)
        st.markdown("""
            <div style="text-align:center;">
                <span class="metric-chip">Transfer Learning</span>
                <span class="metric-chip">Grad-CAM</span>
                <span class="metric-chip">DenseNet121</span>
            </div>
        """, unsafe_allow_html=True)

    # ── Konten Utama ──────────────────────────────────────────
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<p class="section-title">📁 Upload Citra X-Ray</p>', unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Drag & drop atau klik untuk memilih file",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

        if uploaded_file:
            pil_image = Image.open(uploaded_file)
            st.image(pil_image, caption="Citra X-Ray yang Diunggah", use_column_width=True)

            st.markdown(f"""
                <div style="display:flex; gap:8px; margin-top:8px; flex-wrap:wrap;">
                    <span class="metric-chip">📐 {pil_image.width}×{pil_image.height} px</span>
                    <span class="metric-chip">🎨 Mode: {pil_image.mode}</span>
                    <span class="metric-chip">📄 {uploaded_file.name}</span>
                </div>
            """, unsafe_allow_html=True)

            run = st.button("🔍 Mulai Klasifikasi")
        else:
            st.markdown("""
                <div style="border:2px dashed #90caf9; border-radius:12px; padding:50px 20px;
                            text-align:center; background:#f0f7ff; color:#90a4ae;">
                    <span style="font-size:3rem;">🩻</span><br><br>
                    <p style="margin:0; font-size:0.95rem; font-weight:500; color:#64748b;">
                        Upload citra X-Ray tulang belakang
                    </p>
                    <p style="margin:4px 0 0 0; font-size:0.8rem; color:#94a3b8;">
                        Format: JPG, JPEG, PNG
                    </p>
                </div>
            """, unsafe_allow_html=True)
            run = False

    with col2:
        st.markdown('<p class="section-title">📊 Hasil Analisis</p>', unsafe_allow_html=True)

        if uploaded_file and run:
            with st.spinner("🔄 Memuat model DenseNet121..."):
                model = load_model()

            if model is None:
                st.error("❌ Model gagal dimuat.")
                return

            with st.spinner("🧠 Menganalisis citra X-Ray..."):
                img_tensor              = preprocess_image(pil_image)
                label, confidence, prob = predict(model, img_tensor)

            show_result(label, confidence, prob)

            if show_gradcam:
                st.markdown("""
                    <p class="section-title" style="margin-top:20px;">🔥 Grad-CAM Visualization</p>
                """, unsafe_allow_html=True)

                with st.spinner("Menghasilkan peta aktivasi..."):
                    heatmap = make_gradcam_heatmap(img_tensor, model, prob)

                if heatmap is not None:
                    overlay = overlay_gradcam(pil_image, heatmap)
                    g1, g2  = st.columns(2)
                    with g1:
                        fig, ax = plt.subplots(figsize=(4, 4))
                        fig.patch.set_facecolor('#0d1117')
                        ax.set_facecolor('#0d1117')
                        im = ax.imshow(heatmap, cmap="jet")
                        ax.axis("off")
                        ax.set_title("Activation Heatmap", fontsize=9,
                                     color='white', pad=8)
                        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
                    with g2:
                        st.image(overlay, caption="Overlay pada Gambar Asli",
                                 use_column_width=True)

                    st.markdown("""
                        <div style="background:#fff8e1; border-left:3px solid #f9a825;
                                    border-radius:6px; padding:8px 12px; margin-top:8px;">
                            <p style="margin:0; font-size:0.78rem; color:#795548;">
                                🔴 <strong>Merah/Kuning</strong> = Area dengan aktivasi tinggi (fokus model)<br>
                                🔵 <strong>Biru</strong> = Area dengan aktivasi rendah
                            </p>
                        </div>
                    """, unsafe_allow_html=True)

            with st.expander("📋 Detail Prediksi Lengkap"):
                st.markdown(f"""
                | Parameter | Nilai |
                |---|---|
                | Arsitektur | `DenseNet121` |
                | File Model | `best_densenet121_e4.keras` |
                | Ukuran Input | `224 × 224 px` |
                | Probabilitas Scoliosis | `{prob:.6f}` |
                | Probabilitas Normal | `{1-prob:.6f}` |
                | Threshold | `{THRESHOLD}` |
                | Hasil Prediksi | **{label}** |
                | Kepercayaan | **{confidence:.2f}%** |
                """)

        elif not uploaded_file:
            st.markdown("""
                <div style="border:1px solid #e2e8f0; border-radius:12px; padding:50px 20px;
                            text-align:center; background:#fafafa; color:#94a3b8; margin-top:0;">
                    <span style="font-size:3rem; opacity:0.4;">📊</span><br><br>
                    <p style="margin:0; font-size:0.95rem; font-weight:500; color:#94a3b8;">
                        Hasil prediksi akan muncul di sini
                    </p>
                    <p style="margin:4px 0 0 0; font-size:0.8rem; color:#cbd5e1;">
                        Upload gambar dan tekan <strong>Mulai Klasifikasi</strong>
                    </p>
                </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
