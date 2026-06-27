"""
Aplikasi Streamlit: Klasifikasi Skoliosis pada Citra X-Ray Tulang Belakang
Menggunakan Transfer Learning dengan Arsitektur DenseNet121
"""

import os
import sys
import requests
import streamlit as st

# ─────────────────────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────────────────────
IMG_SIZE          = (224, 224)
THRESHOLD         = 0.50
POSITIVE_CLASS    = "Scoliosis"
NEGATIVE_CLASS    = "Normal"
GDRIVE_ID         = "1n3JdcdVfqYFNlYGVeywipspexElt8QPG"
LOCAL_MODEL_PATH  = "models/best_densenet121_e4.keras"
CACHED_MODEL_PATH = "/tmp/best_densenet121_e4.keras"
LAST_CONV_LAYER   = "conv5_block16_concat"

# ─────────────────────────────────────────────────────────────
# CEK DEPENDENCIES (lazy, dengan pesan error jelas)
# ─────────────────────────────────────────────────────────────

def check_dependencies():
    missing = []
    for pkg in ["numpy", "PIL", "matplotlib", "keras"]:
        try:
            __import__(pkg if pkg != "PIL" else "PIL.Image")
        except ImportError:
            missing.append(pkg)
    return missing

missing = check_dependencies()
if missing:
    st.error(f"❌ Package berikut tidak terinstall: {', '.join(missing)}")
    st.info("Pastikan `requirements.txt` sudah benar dan reboot app.")
    st.stop()

# Import setelah cek
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

os.environ["KERAS_BACKEND"] = "jax"
try:
    import keras
except Exception as e:
    st.error(f"❌ Gagal import keras: {e}")
    st.stop()

# ─────────────────────────────────────────────────────────────
# DOWNLOAD MODEL
# ─────────────────────────────────────────────────────────────

def download_from_gdrive(file_id: str, dest_path: str) -> bool:
    try:
        session  = requests.Session()
        url      = "https://drive.google.com/uc"
        response = session.get(url, params={"id": file_id, "export": "download"}, stream=True)

        token = None
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                token = value
                break

        if token:
            response = session.get(
                url,
                params={"id": file_id, "export": "download", "confirm": token},
                stream=True,
            )

        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=32768):
                if chunk:
                    f.write(chunk)

        return os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000

    except Exception as e:
        st.error(f"Gagal mengunduh model: {e}")
        return False


def resolve_model_path():
    if os.path.exists(LOCAL_MODEL_PATH):
        return LOCAL_MODEL_PATH
    if os.path.exists(CACHED_MODEL_PATH) and os.path.getsize(CACHED_MODEL_PATH) > 1000:
        return CACHED_MODEL_PATH
    with st.spinner("⏬ Mengunduh bobot model dari Google Drive (±50MB)..."):
        success = download_from_gdrive(GDRIVE_ID, CACHED_MODEL_PATH)
    return CACHED_MODEL_PATH if success else None


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
    color    = "#d32f2f" if label == POSITIVE_CLASS else "#2e7d32"
    bg_color = "#fff5f5" if label == POSITIVE_CLASS else "#f1f8e9"
    icon     = "🔴" if label == POSITIVE_CLASS else "🟢"
    st.markdown(f"""
        <div style="background:{bg_color}; border-left:5px solid {color};
                    border-radius:8px; padding:18px 24px; margin-top:10px;">
            <h3 style="margin:0; color:{color};">{icon} {label}</h3>
            <p style="margin:6px 0 4px 0; font-size:0.95rem; color:#555;">
                Kepercayaan model: <strong>{confidence:.2f}%</strong>
            </p>
            <div style="background:#e0e0e0;border-radius:8px;height:18px;width:100%;margin-bottom:14px;">
                <div style="background:{color};width:{confidence:.1f}%;height:18px;border-radius:8px;"></div>
            </div>
            <p style="margin:0; font-size:0.82rem; color:#888;">
                Prob. scoliosis: {prob:.4f} &nbsp;|&nbsp; Threshold: {THRESHOLD}
            </p>
        </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="Klasifikasi Skoliosis X-Ray", page_icon="🩻", layout="wide")

    st.markdown("""
        <style>
            .block-container { padding-top:1.5rem; }
            .stButton > button {
                width:100%; background-color:#1565c0; color:white;
                border:none; border-radius:8px; padding:10px 0;
                font-size:1rem; font-weight:600; margin-top:8px;
            }
            .stButton > button:hover { background-color:#0d47a1; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div style="text-align:center; padding:10px 0 20px 0;">
            <h1>🩻 KLASIFIKASI SKOLIOSIS PADA CITRA X-RAY TULANG BELAKANG</h1>
            <p style="font-size:1rem; color:#555; max-width:720px; margin:0 auto;">
                Menggunakan <strong>Transfer Learning</strong> dengan arsitektur
                <strong>DenseNet121</strong> untuk mendeteksi kondisi skoliosis.
            </p>
            <hr style="border:none; border-top:1px solid #e0e0e0; margin:16px auto; width:60%;">
        </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/6/68/X-ray_of_a_lumbar_spine.jpg/220px-X-ray_of_a_lumbar_spine.jpg",
            caption="Contoh X-Ray Tulang Belakang", use_column_width=True,
        )
        st.markdown("## ⚙️ Info Model")
        st.info("**Model:** DenseNet121\n\n**File:** best_densenet121_e4.keras\n\n**Input:** 224×224 px\n\n**Threshold:** 0.50")
        show_gradcam = st.checkbox("Tampilkan Grad-CAM", value=True)
        st.markdown("---")
        st.markdown("### ℹ️ Kelas")
        st.success("🟢 **Normal**")
        st.error("🔴 **Scoliosis**")
        st.markdown("---")
        st.markdown("<small style='color:#888;'>Skripsi S1 Sains Data<br>UPN Veteran Jawa Timur</small>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### 📁 Upload Citra X-Ray")
        uploaded_file = st.file_uploader("Pilih file (JPG/PNG)", type=["jpg","jpeg","png"], label_visibility="collapsed")
        if uploaded_file:
            pil_image = Image.open(uploaded_file)
            st.image(pil_image, caption="Gambar yang Diunggah", use_column_width=True)
            st.caption(f"Dimensi: {pil_image.width}×{pil_image.height} px | Mode: {pil_image.mode}")
            run = st.button("🔍 Klasifikasi")
        else:
            st.info("Silakan upload citra X-Ray tulang belakang.")
            run = False

    with col2:
        st.markdown("### 📊 Hasil Prediksi")
        if uploaded_file and run:
            with st.spinner("Memuat model..."):
                model = load_model()
            if model is None:
                st.error("❌ Model gagal dimuat.")
                return

            with st.spinner("Menganalisis citra..."):
                img_tensor             = preprocess_image(pil_image)
                label, confidence, prob = predict(model, img_tensor)

            show_result(label, confidence, prob)

            if show_gradcam:
                st.markdown("#### 🔥 Grad-CAM")
                with st.spinner("Menghasilkan peta aktivasi..."):
                    heatmap = make_gradcam_heatmap(img_tensor, model, prob)
                if heatmap is not None:
                    overlay = overlay_gradcam(pil_image, heatmap)
                    g1, g2  = st.columns(2)
                    with g1:
                        fig, ax = plt.subplots(figsize=(4, 4))
                        ax.imshow(heatmap, cmap="jet")
                        ax.axis("off")
                        ax.set_title("Heatmap", fontsize=10)
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
                    with g2:
                        st.image(overlay, caption="Overlay", use_column_width=True)
                    st.caption("Merah = area fokus model tertinggi.")

            with st.expander("📋 Detail Prediksi"):
                st.markdown(f"""
                | Parameter | Nilai |
                |---|---|
                | Arsitektur | `DenseNet121` |
                | Prob. Scoliosis | `{prob:.6f}` |
                | Prob. Normal | `{1-prob:.6f}` |
                | Threshold | `{THRESHOLD}` |
                | Prediksi | **{label}** |
                | Kepercayaan | **{confidence:.2f}%** |
                """)
        elif not uploaded_file:
            st.markdown("""
                <div style="background:#f5f5f5;border-radius:8px;padding:40px;
                            text-align:center;color:#9e9e9e;margin-top:20px;">
                    <span style="font-size:3rem;">🩻</span><br><br>
                    Upload gambar dan tekan <strong>Klasifikasi</strong>.
                </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
