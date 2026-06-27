"""
Aplikasi Streamlit: Klasifikasi Skoliosis pada Citra X-Ray Tulang Belakang
Menggunakan Transfer Learning dengan Arsitektur DenseNet121
"""

import os
import gdown
import numpy as np
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from PIL import Image

# ─────────────────────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────────────────────
IMG_SIZE       = (224, 224)
THRESHOLD      = 0.50
POSITIVE_CLASS = "Scoliosis"
NEGATIVE_CLASS = "Normal"

# ── Bobot model DenseNet121 terbaik ──────────────────────────
GDRIVE_ID          = "1n3JdcdVfqYFNlYGVeywipspexElt8QPG"
LOCAL_MODEL_PATH   = "models/best_densenet121_e4.keras"
CACHED_MODEL_PATH  = "/tmp/best_densenet121_e4.keras"
LAST_CONV_LAYER    = "conv5_block16_concat"

# ─────────────────────────────────────────────────────────────
# UTILITAS GOOGLE DRIVE
# ─────────────────────────────────────────────────────────────

def download_from_gdrive(file_id: str, dest_path: str) -> bool:
    """Unduh file dari Google Drive menggunakan gdown."""
    try:
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, dest_path, quiet=False)
        return os.path.exists(dest_path)
    except Exception as e:
        st.error(f"Gagal mengunduh dari Google Drive: {e}")
        return False


def resolve_model_path() -> str | None:
    """
    Prioritas:
    1. Path lokal (jika file sudah ada di sistem)
    2. Cache /tmp (jika sudah pernah diunduh)
    3. Unduh dari Google Drive
    """
    if os.path.exists(LOCAL_MODEL_PATH):
        return LOCAL_MODEL_PATH

    if os.path.exists(CACHED_MODEL_PATH):
        return CACHED_MODEL_PATH

    with st.spinner("Mengunduh bobot model DenseNet121 dari Google Drive..."):
        success = download_from_gdrive(GDRIVE_ID, CACHED_MODEL_PATH)

    return CACHED_MODEL_PATH if success else None


# ─────────────────────────────────────────────────────────────
# PEMUATAN MODEL
# ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model():
    model_path = resolve_model_path()
    if model_path is None:
        return None
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None


# ─────────────────────────────────────────────────────────────
# PRA-PROSES GAMBAR
# ─────────────────────────────────────────────────────────────

def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """Ubah gambar PIL menjadi tensor input model (1, 224, 224, 3)."""
    img = pil_image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


# ─────────────────────────────────────────────────────────────
# PREDIKSI
# ─────────────────────────────────────────────────────────────

def predict(model, img_tensor: np.ndarray):
    """
    Kembalikan (label, confidence_persen, raw_prob).
    Model sigmoid → 1 neuron output (probabilitas scoliosis).
    """
    prob = float(model.predict(img_tensor, verbose=0)[0][0])
    if prob >= THRESHOLD:
        label      = POSITIVE_CLASS
        confidence = prob * 100
    else:
        label      = NEGATIVE_CLASS
        confidence = (1 - prob) * 100
    return label, confidence, prob


# ─────────────────────────────────────────────────────────────
# GRAD-CAM
# ─────────────────────────────────────────────────────────────

def make_gradcam_heatmap(img_tensor: np.ndarray, model, prob: float) -> np.ndarray | None:
    """Hasilkan heatmap Grad-CAM sebagai array numpy."""
    try:
        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[
                model.get_layer(LAST_CONV_LAYER).output,
                model.output,
            ],
        )

        with tf.GradientTape() as tape:
            conv_output, prediction = grad_model(img_tensor, training=False)
            probability = prediction[:, 0]
            loss = probability if prob >= THRESHOLD else (1 - probability)

        grads        = tape.gradient(loss, conv_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_output  = conv_output[0]
        heatmap      = tf.reduce_sum(conv_output * pooled_grads, axis=-1).numpy()

        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        return heatmap

    except Exception as e:
        st.warning(f"Grad-CAM tidak tersedia: {e}")
        return None


def overlay_gradcam(pil_image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45):
    """Tumpangkan heatmap Grad-CAM ke gambar asli."""
    heatmap_resized = np.uint8(255 * heatmap)
    jet             = cm.get_cmap("jet")
    jet_colors      = jet(np.arange(256))[:, :3]
    jet_heatmap     = jet_colors[heatmap_resized]
    jet_heatmap     = Image.fromarray(np.uint8(jet_heatmap * 255)).resize(
        pil_image.size, Image.BILINEAR
    )
    original_rgb  = pil_image.convert("RGB")
    superimposed  = Image.blend(original_rgb, jet_heatmap, alpha)
    return superimposed


# ─────────────────────────────────────────────────────────────
# TAMPILAN HASIL
# ─────────────────────────────────────────────────────────────

def show_result(label: str, confidence: float, prob: float):
    """Render kartu hasil prediksi."""
    color    = "#d32f2f" if label == POSITIVE_CLASS else "#2e7d32"
    bg_color = "#fff5f5" if label == POSITIVE_CLASS else "#f1f8e9"
    icon     = "🔴" if label == POSITIVE_CLASS else "🟢"

    bar_fill = f"""
    <div style="background:#e0e0e0; border-radius:8px; height:18px; width:100%; margin:6px 0 14px 0;">
        <div style="background:{color}; width:{confidence:.1f}%; height:18px;
                    border-radius:8px; transition:width .5s;"></div>
    </div>
    """

    st.markdown(
        f"""
        <div style="background:{bg_color}; border-left:5px solid {color};
                    border-radius:8px; padding:18px 24px; margin-top:10px;">
            <h3 style="margin:0; color:{color};">{icon} {label}</h3>
            <p style="margin:6px 0 0 0; font-size:0.95rem; color:#555;">
                Kepercayaan model: <strong>{confidence:.2f}%</strong>
            </p>
            {bar_fill}
            <p style="margin:0; font-size:0.82rem; color:#888;">
                Probabilitas raw scoliosis: {prob:.4f} &nbsp;|&nbsp; Threshold: {THRESHOLD}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Klasifikasi Skoliosis X-Ray",
        page_icon="🩻",
        layout="wide",
    )

    st.markdown(
        """
        <style>
            .block-container { padding-top: 1.5rem; }
            h1 { line-height: 1.25 !important; }
            .stButton > button {
                width: 100%;
                background-color: #1565c0;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 0;
                font-size: 1rem;
                font-weight: 600;
                margin-top: 8px;
            }
            .stButton > button:hover { background-color: #0d47a1; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ── Header ────────────────────────────────────────────────
    st.markdown(
        """
        <div style="text-align:center; padding: 10px 0 20px 0;">
            <h1>🩻 KLASIFIKASI SKOLIOSIS PADA CITRA X-RAY TULANG BELAKANG</h1>
            <p style="font-size:1rem; color:#555; max-width:720px; margin:0 auto;">
                Menggunakan <strong>Transfer Learning</strong> dengan arsitektur
                <strong>DenseNet121</strong> yang dilatih pada dataset X-Ray tulang
                belakang untuk mendeteksi kondisi skoliosis.
            </p>
            <hr style="border:none; border-top:1px solid #e0e0e0; margin:16px auto; width:60%;">
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Sidebar ───────────────────────────────────────────────
    with st.sidebar:
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/6/68/"
            "X-ray_of_a_lumbar_spine.jpg/220px-X-ray_of_a_lumbar_spine.jpg",
            caption="Contoh X-Ray Tulang Belakang",
            use_column_width=True,
        )

        st.markdown("## ⚙️ Konfigurasi Model")
        st.info(
            "**Model:** DenseNet121\n\n"
            "**File:** best_densenet121_e4.keras\n\n"
            "**Input:** 224 × 224 px\n\n"
            "**Threshold:** 0.50"
        )

        show_gradcam = st.checkbox(
            "Tampilkan Grad-CAM",
            value=True,
            help="Visualisasi area yang menjadi fokus model saat membuat prediksi.",
        )

        st.markdown("---")
        st.markdown("### ℹ️ Informasi Kelas")
        st.success("🟢 **Normal** — Tulang belakang tidak menunjukkan tanda skoliosis.")
        st.error("🔴 **Scoliosis** — Terdeteksi kelengkungan lateral pada tulang belakang.")

        st.markdown("---")
        st.markdown(
            "<small style='color:#888;'>Dibuat untuk skripsi S1 Sains Data,<br>"
            "UPN Veteran Jawa Timur</small>",
            unsafe_allow_html=True,
        )

    # ── Konten utama ──────────────────────────────────────────
    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown("### 📁 Upload Citra X-Ray")
        uploaded_file = st.file_uploader(
            "Pilih file gambar (JPG / PNG)",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

        if uploaded_file:
            pil_image = Image.open(uploaded_file)
            st.image(pil_image, caption="Gambar yang Diunggah", use_column_width=True)
            st.caption(
                f"Dimensi: {pil_image.width} × {pil_image.height} px  |  "
                f"Mode: {pil_image.mode}"
            )
            run = st.button("🔍 Klasifikasi")
        else:
            st.info("Silakan upload citra X-Ray tulang belakang untuk memulai.")
            run = False

    with col_result:
        st.markdown("### 📊 Hasil Prediksi")

        if uploaded_file and run:
            # Muat model
            with st.spinner("Memuat model DenseNet121..."):
                model = load_model()

            if model is None:
                st.error(
                    "❌ Model gagal dimuat. Pastikan koneksi internet aktif "
                    "atau file model tersedia."
                )
                return

            # Pra-proses & prediksi
            with st.spinner("Menganalisis citra..."):
                img_tensor = preprocess_image(pil_image)
                label, confidence, prob = predict(model, img_tensor)

            # Kartu hasil
            show_result(label, confidence, prob)

            # Grad-CAM
            if show_gradcam:
                st.markdown("#### 🔥 Visualisasi Grad-CAM")
                with st.spinner("Menghasilkan peta aktivasi..."):
                    heatmap = make_gradcam_heatmap(img_tensor, model, prob)

                if heatmap is not None:
                    overlay = overlay_gradcam(pil_image, heatmap)

                    g1, g2 = st.columns(2)
                    with g1:
                        fig, ax = plt.subplots(figsize=(4, 4))
                        ax.imshow(heatmap, cmap="jet")
                        ax.axis("off")
                        ax.set_title("Heatmap", fontsize=10)
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
                    with g2:
                        st.image(
                            overlay,
                            caption="Overlay pada Gambar Asli",
                            use_column_width=True,
                        )

                    st.caption(
                        "Grad-CAM menampilkan area yang paling berpengaruh pada "
                        "keputusan model. Warna merah = aktivasi tinggi."
                    )

            # Detail prediksi
            with st.expander("📋 Detail Prediksi"):
                st.markdown(
                    f"""
                    | Parameter | Nilai |
                    |---|---|
                    | Arsitektur | `DenseNet121` |
                    | File Model | `best_densenet121_e4.keras` |
                    | Ukuran Input | `224 × 224 px` |
                    | Probabilitas Scoliosis | `{prob:.6f}` |
                    | Probabilitas Normal | `{1 - prob:.6f}` |
                    | Threshold | `{THRESHOLD}` |
                    | Kelas Prediksi | **{label}** |
                    | Kepercayaan | **{confidence:.2f}%** |
                    """
                )

        elif not uploaded_file:
            st.markdown(
                """
                <div style="background:#f5f5f5; border-radius:8px; padding:40px;
                            text-align:center; color:#9e9e9e; margin-top:20px;">
                    <span style="font-size:3rem;">🩻</span><br><br>
                    Hasil prediksi akan muncul di sini setelah gambar diunggah
                    dan tombol <strong>Klasifikasi</strong> ditekan.
                </div>
                """,
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
