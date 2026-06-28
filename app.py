"""
Aplikasi Streamlit: Klasifikasi Skoliosis pada Citra X-Ray Tulang Belakang
Menggunakan Transfer Learning dengan Arsitektur DenseNet121

Fix revisi:
1. Gambar di-cap max-height agar tidak memanjang ke bawah
2. Hasil prediksi persistent via session_state (tidak hilang saat klik Grad-CAM)
3. Heatmap + Overlay langsung tampil berdampingan tanpa perlu klik tab
4. Gambar Grad-CAM di-cap max-height agar tidak memanjang
"""

import os
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import io

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
    """Render heatmap matplotlib ke bytes PNG — tidak perlu st.pyplot agar tidak rerun."""
    fig, ax = plt.subplots(figsize=(3, 3), dpi=120)
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

# ─────────────────────────────────────────────────────────────
# RENDER HASIL PREDIKSI — fungsi terpisah agar bisa dipanggil
# dari dua tempat (setelah klasifikasi & setelah Grad-CAM)
# ─────────────────────────────────────────────────────────────

def render_hasil(label, confidence, prob):
    is_sc  = label == POSITIVE_CLASS
    color  = "#c62828" if is_sc else "#2e7d32"
    bg     = "#fce4e4" if is_sc else "#e8f5e9"
    border = "#ef9a9a" if is_sc else "#a5d6a7"
    icon   = "🔴" if is_sc else "🟢"
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
                <div style="background:{color}; width:{confidence:.1f}%;
                            height:10px; border-radius:6px;"></div>
            </div>
            <p style="margin:10px 0 0 0; font-size:0.72rem; color:#64748b;">
                Prob. Scoliosis: <strong>{prob:.4f}</strong><br>
                Prob. Normal: <strong>{1 - prob:.4f}</strong><br>
                Threshold: <strong>{THRESHOLD}</strong>
            </p>
        </div>
    """, unsafe_allow_html=True)

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

            .block-container {
                padding: 0 1.5rem 0.5rem 1.5rem !important;
                max-width: 100% !important;
            }
            [data-testid="stAppViewContainer"] { background: #f0f4f8; }
            [data-testid="stHeader"] { display: none; }

            .app-header {
                background: linear-gradient(135deg, #0d47a1 0%, #1976d2 100%);
                color: white; padding: 12px 24px;
                border-radius: 0 0 12px 12px; margin-bottom: 12px;
                display: flex; align-items: center; gap: 14px;
                box-shadow: 0 3px 12px rgba(13,71,161,0.25);
            }
            .app-header h1 { margin: 0; font-size: 1.05rem; font-weight: 700; line-height: 1.3; }
            .app-header p  { margin: 0; font-size: 0.72rem; opacity: 0.8; }

            .panel-title { font-size: 0.65rem; font-weight: 700; text-transform: uppercase;
                           letter-spacing: 1.2px; color: #64748b; margin-bottom: 8px; }

            [data-testid="stFileUploader"] > div {
                border: 2px dashed #90caf9 !important;
                border-radius: 10px !important;
                background: #f0f7ff !important;
                padding: 6px !important;
            }

            .stButton > button {
                width: 100%;
                background: linear-gradient(135deg, #1565c0, #1976d2);
                color: white; border: none; border-radius: 8px;
                padding: 9px 0; font-size: 0.9rem; font-weight: 600;
                margin-top: 6px; box-shadow: 0 2px 8px rgba(21,101,192,0.3);
            }
            .stButton > button:hover {
                background: linear-gradient(135deg, #0d47a1, #1565c0);
                transform: translateY(-1px);
            }

            .chip {
                display: inline-block; background: #e8f0fe; color: #1a56db;
                padding: 2px 8px; border-radius: 12px;
                font-size: 0.7rem; font-weight: 600; margin: 1px;
            }

            /* FIX 1 & 4: Cap tinggi gambar agar tidak memanjang ke bawah */
            [data-testid="stImage"] img {
                max-height: 300px !important;
                width: 100% !important;
                object-fit: contain !important;
                border-radius: 8px;
            }

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

    # ── Layout 3 kolom ────────────────────────────────────────
    col_upload, col_result, col_gradcam = st.columns([1, 1, 1], gap="medium")

    # ── Kolom 1: Upload ───────────────────────────────────────
    with col_upload:
        st.markdown('<p class="panel-title">📁 Upload Citra X-Ray</p>', unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Upload", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
        )

        if uploaded_file:
            pil_image = Image.open(uploaded_file)
            # FIX 1: gambar dibatasi max-height via CSS di atas
            st.image(pil_image, use_column_width=True)
            st.markdown(f"""
                <div style="margin-top:4px;">
                    <span class="chip">📐 {pil_image.width}×{pil_image.height}px</span>
                    <span class="chip">🎨 {pil_image.mode}</span>
                </div>
            """, unsafe_allow_html=True)
            run = st.button("🔍 Klasifikasi")
        else:
            pil_image = None
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

        st.markdown("""
            <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px;
                        padding:10px 12px; margin-top:10px; font-size:0.75rem; color:#475569;">
                <p style="margin:0; font-weight:700; color:#1e293b; margin-bottom:4px;">⚙️ Info Model</p>
                <p style="margin:0;">🏗 DenseNet121 (Keras + TF-CPU)</p>
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

    # ── Kolom 2: Hasil Prediksi ───────────────────────────────
    with col_result:
        st.markdown('<p class="panel-title">📊 Hasil Prediksi</p>', unsafe_allow_html=True)

        # FIX 2: Jalankan klasifikasi dan simpan ke session_state
        if uploaded_file and run:
            with st.spinner("Memuat model…"):
                model = load_model()
            if model is None:
                st.error("❌ Model gagal dimuat.")
            else:
                with st.spinner("Menganalisis gambar…"):
                    img_tensor              = preprocess_image(pil_image)
                    label, confidence, prob = predict(model, img_tensor)

                # Simpan semua hasil ke session_state
                st.session_state["label"]      = label
                st.session_state["confidence"] = confidence
                st.session_state["prob"]       = prob
                st.session_state["img_tensor"] = img_tensor
                st.session_state["pil_image"]  = pil_image
                st.session_state["model"]      = model
                st.session_state["ran"]        = True
                # Reset Grad-CAM lama jika gambar baru diklasifikasi
                st.session_state["gradcam_done"] = False

        # FIX 2: Tampilkan dari session_state — tidak akan hilang saat rerun
        if st.session_state.get("ran"):
            render_hasil(
                st.session_state["label"],
                st.session_state["confidence"],
                st.session_state["prob"],
            )
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

        if st.session_state.get("ran") and uploaded_file:
            if st.button("🔥 Generate Grad-CAM"):
                with st.spinner("Menghasilkan peta aktivasi…"):
                    heatmap = make_gradcam_heatmap(
                        st.session_state["img_tensor"],
                        st.session_state["model"],
                        st.session_state["prob"],
                    )
                if heatmap is not None:
                    # FIX 3: Pre-render keduanya sekaligus ke bytes,
                    # simpan di session_state agar tidak perlu klik tab
                    overlay = make_overlay(st.session_state["pil_image"], heatmap)

                    # Simpan heatmap sebagai PNG bytes
                    st.session_state["heatmap_bytes"] = heatmap_to_png_bytes(heatmap)

                    # Simpan overlay sebagai PNG bytes
                    buf = io.BytesIO()
                    overlay.save(buf, format="PNG")
                    st.session_state["overlay_bytes"] = buf.getvalue()

                    st.session_state["gradcam_done"] = True

            # FIX 3 & 4: Tampilkan heatmap + overlay berdampingan langsung tanpa tab
            if st.session_state.get("gradcam_done"):
                gc1, gc2 = st.columns(2, gap="small")
                with gc1:
                    st.markdown("""
                        <p style="font-size:0.68rem; font-weight:700; color:#64748b;
                                   text-align:center; margin-bottom:4px;">🌡 HEATMAP</p>
                    """, unsafe_allow_html=True)
                    # FIX 4: gambar dibatasi max-height via CSS global
                    st.image(st.session_state["heatmap_bytes"], use_column_width=True)
                with gc2:
                    st.markdown("""
                        <p style="font-size:0.68rem; font-weight:700; color:#64748b;
                                   text-align:center; margin-bottom:4px;">🖼 OVERLAY</p>
                    """, unsafe_allow_html=True)
                    st.image(st.session_state["overlay_bytes"], use_column_width=True)

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
                        Klik "Klasifikasi" dulu, lalu generate Grad-CAM
                    </p>
                </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
