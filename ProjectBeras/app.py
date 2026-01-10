import streamlit as st
import cv2
import numpy as np
import joblib
import pandas as pd
from PIL import Image
from datetime import datetime
import os

st.set_page_config(
    page_title="Rice Quality - Ipsala",
    page_icon="🌾",
    layout="wide"
)

if 'history' not in st.session_state:
    st.session_state['history'] = []

@st.cache_resource
def load_resources():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        path_model = os.path.join(base_dir, 'model_beras_xgboost_final.pkl')
        path_scaler = os.path.join(base_dir, 'scaler_beras.pkl')
        path_le = os.path.join(base_dir, 'label_encoder_beras.pkl')
        model = joblib.load(path_model)
        scaler = joblib.load(path_scaler)
        le = joblib.load(path_le)
        return model, scaler, le
    except:
        return None, None, None

model, scaler, le = load_resources()

def extract_feature_from_contour(cnt, img_gray, img_rgb):
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / h if h != 0 else 0
    roundness = (4 * np.pi * area) / (perimeter ** 2) if perimeter != 0 else 0
    mask = np.zeros(img_gray.shape, np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    mean_val = cv2.mean(img_rgb, mask=mask)
    features = [area, perimeter, aspect_ratio, roundness, mean_val[0], mean_val[1], mean_val[2]]
    return features, (x, y, w, h)

st.sidebar.title("Rice Quality")
st.sidebar.caption("Sistem Klasifikasi Mutu Beras (Ipsala)")
st.sidebar.markdown("---")
menu = st.sidebar.radio("Navigasi", ["Aplikasi Utama", "Riwayat Analisis"])
st.sidebar.markdown("---")

with st.sidebar.expander("ℹ️ Spesifikasi Model", expanded=True):
    st.write("**Algoritma:** XGBoost Classifier")
    st.write("**Fitur:** Morfologi & Warna")
    st.write("**Fitur Spesial:** Adaptive Scaling (Anti-Zoom)")

st.sidebar.markdown("---")
st.sidebar.info("© 2026 - Data Science")

if menu == "Aplikasi Utama":
    st.title("🌾 Klasifikasi Mutu Beras (Ipsala)")
    st.write("Upload citra beras (20-100 butir). Sistem otomatis mengkalibrasi ukuran (Auto-Scaling).")

    if model is None:
        st.error("File model tidak ditemukan.")
        st.stop()

    uploaded_file = st.file_uploader("Pilih Gambar", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        col_img, col_res = st.columns(2)
        image = Image.open(uploaded_file)
        base_width = 1000
        w_percent = base_width / float(image.size[0])
        h_size = int(float(image.size[1]) * w_percent)
        image_display = image.resize((base_width, h_size), Image.Resampling.LANCZOS)

        with col_img:
            st.subheader("📸 Citra Input")
            st.image(image_display, use_container_width=True)

        img_array = np.array(image.convert('RGB'))
        max_dim = 2000
        if max(img_array.shape) > max_dim:
            scale_img = max_dim / max(img_array.shape)
            img_array = cv2.resize(img_array, None, fx=scale_img, fy=scale_img)

        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid_areas = []
        temp_features = []
        AVG_AREA_TRAINING = 12000

        max_area_in_image = max([cv2.contourArea(c) for c in contours]) if contours else 0
        dynamic_threshold = max(100, max_area_in_image * 0.01)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > dynamic_threshold:
                features_raw, bbox = extract_feature_from_contour(cnt, img_gray, img_array)
                valid_areas.append(area)
                temp_features.append({'cnt': cnt, 'feat': features_raw, 'bbox': bbox})

        scale_factor = AVG_AREA_TRAINING / np.mean(valid_areas) if valid_areas else 1.0

        results_label = []
        results_data = []
        img_result = img_array.copy()
        count_obj = 0

        for item in temp_features:
            count_obj += 1
            cnt = item['cnt']
            feat_raw = item['feat']
            x, y, w, h = item['bbox']

            area_corrected = feat_raw[0] * scale_factor
            perimeter_corrected = feat_raw[1] * np.sqrt(scale_factor)

            features_corrected = [
                area_corrected,
                perimeter_corrected,
                feat_raw[2],
                feat_raw[3],
                feat_raw[4],
                feat_raw[5],
                feat_raw[6]
            ]

            cols = ['Area', 'Perimeter', 'AspectRatio', 'Roundness', 'Red_Mean', 'Green_Mean', 'Blue_Mean']
            features_df = pd.DataFrame([features_corrected], columns=cols)
            features_scaled = scaler.transform(features_df)

            prediction_idx = model.predict(features_scaled)[0]
            label_name = le.inverse_transform([prediction_idx])[0]

            rect = cv2.minAreaRect(cnt)
            (_, _), (dim1, dim2), _ = rect
            real_aspect_ratio = max(dim1, dim2) / min(dim1, dim2) if min(dim1, dim2) > 0 else 0

            if label_name == 'Broken' and real_aspect_ratio > 2.0:
                label_name = 'Whole'
            elif real_aspect_ratio < 1.5:
                label_name = 'Broken'

            results_label.append(label_name)
            row = feat_raw.copy()
            row.insert(0, count_obj)
            row.append(label_name)
            results_data.append(row)

            color_map = {
                'Whole': (0, 255, 0),
                'Broken': (255, 0, 0),
                'Chalky': (255, 0, 255),
                'Discolored': (255, 255, 0)
            }

            color = color_map.get(label_name, (255, 255, 255))
            cv2.rectangle(img_result, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img_result, label_name, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        with col_res:
            st.subheader("🔍 Hasil Deteksi")
            st.image(img_result, use_container_width=True,
                     caption=f"Total: {count_obj} | Scale Factor: {scale_factor:.2f}x")

        if results_label:
            st.markdown("---")
            st.subheader("📊 Analisis Mutu")

            df_count = pd.DataFrame(results_label, columns=['Mutu'])
            counts = df_count['Mutu'].value_counts()
            total = len(results_label)
            whole_pct = (counts.get('Whole', 0) / total) * 100 if total > 0 else 0

            status = "Sangat Baik (Premium)" if whole_pct > 80 else \
                     "Cukup Baik (Medium)" if whole_pct > 50 else \
                     "Kurang Baik (Rendah)"

            (st.success if whole_pct > 80 else
             st.warning if whole_pct > 50 else
             st.error)(f"Sampel beras: **{status}** ({whole_pct:.1f}%)")

            df_table = pd.DataFrame(
                results_data,
                columns=['ID', 'Area', 'Perimeter', 'Aspect', 'Roundness', 'R', 'G', 'B', 'Kelas']
            )

            st.dataframe(df_table, use_container_width=True)

elif menu == "Riwayat Analisis":
    st.title("📜 Log Riwayat Analisis")
    if st.session_state['history']:
        st.dataframe(pd.DataFrame(st.session_state['history']), use_container_width=True)
        if st.button("Hapus Semua Riwayat"):
            st.session_state['history'] = []
            st.rerun()
    else:
        st.info("Belum ada data analisis tersimpan.")
