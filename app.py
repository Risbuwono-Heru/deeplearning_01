"""
Aplikasi Streamlit: Klasifikasi Skoliosis pada Citra X-Ray Tulang Belakang
Menggunakan Transfer Learning dengan Arsitektur DenseNet121
"""

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
            gdown.download(f"https://drive.google.com/uc?id={GDRIVE_ID}", CACHED_MODEL_PATH, quiet=False)
        except Exception as e:
            st.error(f"Gagal mengunduh model: {e}")
            return None
    if os.path.exists(CACHED_MODEL_PATH) and os.path.getsize(CACHED_MODEL_PATH) > 10_000:
        return CACHED_MODEL_PATH
    st.error("❌ File model tidak valid.")
    return None

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
    return np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)

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
        import jax, jax.numpy as jnp
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
        heatmap = jnp.sum(conv_output[0] * pooled_grads, axis=-1)
        heatmap = np.array(jnp.maximum(heatmap, 0))
        if heatmap.max() > 0:
            heatmap /= heatmap.max()
        return heatmap
    except Exception as e:
        st.warning(f"Grad-CAM tidak tersedia: {e}")
        return None

def overlay_gradcam(pil_image, heatmap, alpha=0.45):
    colors      = plt.colormaps["jet"](np.arange(256))[:, :3]
    jet_heatmap = Image.fromarray(np.uint8(colors[np.uint8(255 * heatmap)] * 255))
    jet_heatmap = jet_heatmap.resize(pil_image.size, Image.BILINEAR)
    return Image.blend(pil_image.convert("RGB"), jet_heatmap, alpha)

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Scoliosis X-Ray Classifier",
        page_icon="🩻",
        layout="wide",
        initial_sidebar_state="collapsed",  # sidebar disembunyikan default
    )

    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
            html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

            /* Hilangkan semua padding default */
            .block-container {
                padding: 0 1.5rem 0.5rem 1.5rem !important;
                max-width: 100% !important;
            }
            [data-testid="stAppViewContainer"] { background: #f0f4f8; }
            [data-testid="stHeader"] { display: none; }

            /* Header compact */
            .app-header {
                background: linear-gradient(135deg, #0d47a1 0%, #1976d2 100%);
                color: white;
                padding: 12px 24px;
                border-radius: 0 0 12px 12px;
                margin-bottom: 12px;
                display: flex;
                align-items: center;
                gap: 14px;
                box-shadow: 0 3px 12px rgba(13,71,161,0.25);
            }
            .app-header h1 {
                margin: 0;
                font-size: 1.05rem;
                font-weight: 700;
                line-height: 1.3;
            }
            .app-header p {
                margin: 0;
                font-size: 0.72rem;
                opacity: 0.8;
            }

            /* Panel cards */
            .panel {
                background: white;
                border-radius: 12px;
                padding: 14px 16px;
                box-shadow: 0 1px 6px rgba(0,0,0,0.07);
                height: 100%;
            }
            .panel-title {
                font-size: 0.65rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 1.2px;
                color: #64748b;
                margin-bottom: 8px;
            }

            /* Upload area */
            [data-testid="stFileUploader"] > div {
                border: 2px dashed #90caf9 !important;
                border-radius: 10px !important;
                background: #f0f7ff !important;
                padding: 6px !important;
            }

            /* Classify button */
            .stButton > button {
                width: 100%;
                background: linear-gradient(135deg, #1565c0, #1976d2);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 9px 0;
                font-size: 0.9rem;
                font-weight: 600;
                margin-top: 6px;
                box-shadow: 0 2px 8px rgba(21,101,192,0.3);
            }
            .stButton > button:hover {
                background: linear-gradient(135deg, #0d47a1, #1565c0);
                transform: translateY(-1px);
            }

            /* Result card */
            .result-card {
                border-radius: 10px;
                padding: 12px 16px;
                margin-bottom: 10px;
            }
            .result-scoliosis { background:#fce4e4; border-left:4px solid #c62828; }
            .result-normal    { background:#e8f5e9; border-left:4px solid #2e7d32; }

            /* Metric chip */
            .chip {
                display: inline-block;
                background: #e8f0fe;
                color: #1a56db;
                padding: 2px 8px;
                border-radius: 12px;
                font-size: 0.7rem;
                font-weight: 600;
                margin: 1px;
            }

            /* Sembunyikan footer & menu */
            footer { display: none !important; }
            #MainMenu { display: none !important; }
            [data-testid="collapsedControl"] { display: none !important; }
        </style>
    """, unsafe_allow_html=True)

    # ── Header ────────────────────────────────────────────────
    st.markdown("""
        <div class="app-header">
            <span style="font-size:2rem;">🩻</span>
            <div>
                <h1>Klasifikasi Skoliosis pada Citra X-Ray Tulang Belakang</h1>
                <p>Transfer Learning · DenseNet121 · Grad-CAM Visualization · Binary Classification</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # ── Layout 3 kolom: Upload | Hasil | Grad-CAM ─────────────
    col_upload, col_result, col_gradcam = st.columns([1, 1, 1], gap="medium")

    # ── Kolom 1: Upload ───────────────────────────────────────
    with col_upload:
        st.markdown('<p class="panel-title">📁 Upload Citra X-Ray</p>', unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Upload", type=["jpg","jpeg","png"], label_visibility="collapsed"
        )

        if uploaded_file:
            pil_image = Image.open(uploaded_file)
            st.image(pil_image, use_column_width=True)
            st.markdown(f"""
                <div style="margin-top:4px;">
                    <span class="chip">📐 {pil_image.width}×{pil_image.height}px</span>
                    <span class="chip">🎨 {pil_image.mode}</span>
                </div>
            """, unsafe_allow_html=True)
            run = st.button("🔍 Klasifikasi")
        else:
            st.markdown("""
                <div style="border:2px dashed #90caf9; border-radius:10px; padding:40px 10px;
                            text-align:center; background:#f0f7ff;">
                    <span style="font-size:2.5rem;">🩻</span>
                    <p style="margin:8px 0 0 0; font-size:0.82rem; color:#64748b; font-weight:500;">
                        Upload X-Ray tulang belakang
                    </p>
                    <p style="margin:2px 0 0 0; font-size:0.72rem; color:#94a3b8;">JPG / PNG</p>
                </div>
            """, unsafe_allow_html=True)
            run = False

        # Info model di bawah upload
        st.markdown("""
            <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px;
                        padding:10px 12px; margin-top:10px; font-size:0.75rem; color:#475569;">
                <p style="margin:0; font-weight:700; color:#1e293b; margin-bottom:4px;">⚙️ Info Model</p>
                <p style="margin:0;">🏗 DenseNet121</p>
                <p style="margin:0;">📄 best_densenet121_e4.keras</p>
                <p style="margin:0;">📐 224×224 px &nbsp;|&nbsp; 🎯 Threshold: 0.50</p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div style="margin-top:8px;">
                <div style="background:#e8f5e9; border-left:3px solid #2e7d32; border-radius:6px;
                            padding:6px 10px; margin-bottom:5px; font-size:0.75rem;">
                    🟢 <strong>Normal</strong> — Tidak terdeteksi kelengkungan lateral
                </div>
                <div style="background:#fce4e4; border-left:3px solid #c62828; border-radius:6px;
                            padding:6px 10px; font-size:0.75rem;">
                    🔴 <strong>Scoliosis</strong> — Terdeteksi kelengkungan lateral
                </div>
            </div>
        """, unsafe_allow_html=True)

    # ── Kolom 2: Hasil ────────────────────────────────────────
    with col_result:
        st.markdown('<p class="panel-title">📊 Hasil Prediksi</p>', unsafe_allow_html=True)

        if uploaded_file and run:
            with st.spinner("Memuat model..."):
                model = load_model()

            if model is None:
                st.error("❌ Model gagal dimuat.")
            else:
                with st.spinner("Menganalisis..."):
                    img_tensor              = preprocess_image(pil_image)
                    label, confidence, prob = predict(model, img_tensor)

                is_sc    = label == POSITIVE_CLASS
                color    = "#c62828" if is_sc else "#2e7d32"
                bg       = "#fce4e4" if is_sc else "#e8f5e9"
                border   = "#ef9a9a" if is_sc else "#a5d6a7"
                icon     = "🔴" if is_sc else "🟢"

                st.markdown(f"""
                    <div style="background:{bg}; border:1.5px solid {border};
                                border-radius:12px; padding:16px 18px;
                                box-shadow:0 2px 8px rgba(0,0,0,0.06);">
                        <p style="margin:0; font-size:0.65rem; font-weight:700; text-transform:uppercase;
                                   letter-spacing:1px; color:#64748b;">Hasil Klasifikasi</p>
                        <div style="display:flex; align-items:center; justify-content:space-between; margin:6px 0;">
                            <h2 style="margin:0; color:{color}; font-size:1.5rem;">{icon} {label}</h2>
                            <span style="background:{color}; color:white; padding:4px 14px;
                                          border-radius:20px; font-size:0.9rem; font-weight:700;">
                                {confidence:.1f}%
                            </span>
                        </div>
                        <p style="margin:0 0 5px 0; font-size:0.72rem; color:#555; font-weight:600;">
                            Tingkat Kepercayaan
                        </p>
                        <div style="background:rgba(0,0,0,0.08); border-radius:6px; height:10px;">
                            <div style="background:{color}; width:{confidence:.1f}%; height:10px; border-radius:6px;"></div>
                        </div>
                        <p style="margin:10px 0 0 0; font-size:0.72rem; color:#64748b;">
                            Prob. Scoliosis: <strong>{prob:.4f}</strong><br>
                            Prob. Normal: <strong>{1-prob:.4f}</strong><br>
                            Threshold: <strong>{THRESHOLD}</strong>
                        </p>
                    </div>
                """, unsafe_allow_html=True)

                # Store untuk kolom 3
                st.session_state["img_tensor"] = img_tensor
                st.session_state["prob"]       = prob
                st.session_state["model"]      = model
                st.session_state["pil_image"]  = pil_image
                st.session_state["ran"]        = True

        else:
            st.markdown("""
                <div style="border:1px solid #e2e8f0; border-radius:12px; padding:50px 10px;
                            text-align:center; background:#fafafa;">
                    <span style="font-size:2.5rem; opacity:0.3;">📊</span>
                    <p style="margin:8px 0 0 0; font-size:0.82rem; color:#94a3b8;">
                        Hasil akan muncul setelah klasifikasi
                    </p>
                </div>
            """, unsafe_allow_html=True)

    # ── Kolom 3: Grad-CAM ─────────────────────────────────────
    with col_gradcam:
        st.markdown('<p class="panel-title">🔥 Grad-CAM Visualization</p>', unsafe_allow_html=True)

        if st.session_state.get("ran") and uploaded_file and run:
            with st.spinner("Menghasilkan peta aktivasi..."):
                heatmap = make_gradcam_heatmap(
                    st.session_state["img_tensor"],
                    st.session_state["model"],
                    st.session_state["prob"],
                )

            if heatmap is not None:
                overlay = overlay_gradcam(st.session_state["pil_image"], heatmap)

                tab1, tab2 = st.tabs(["🌡 Heatmap", "🖼 Overlay"])
                with tab1:
                    fig, ax = plt.subplots(figsize=(4, 4))
                    fig.patch.set_facecolor('#0d1117')
                    ax.set_facecolor('#0d1117')
                    im = ax.imshow(heatmap, cmap="jet")
                    ax.axis("off")
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                with tab2:
                    st.image(overlay, use_column_width=True)

                st.markdown("""
                    <div style="background:#fff8e1; border-left:3px solid #f9a825;
                                border-radius:6px; padding:7px 10px; margin-top:6px;
                                font-size:0.72rem; color:#795548;">
                        🔴 <strong>Merah/Kuning</strong> = Aktivasi tinggi (fokus model)<br>
                        🔵 <strong>Biru</strong> = Aktivasi rendah
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style="border:1px solid #e2e8f0; border-radius:12px; padding:50px 10px;
                            text-align:center; background:#fafafa;">
                    <span style="font-size:2.5rem; opacity:0.3;">🔥</span>
                    <p style="margin:8px 0 0 0; font-size:0.82rem; color:#94a3b8;">
                        Grad-CAM akan muncul setelah klasifikasi
                    </p>
                </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
