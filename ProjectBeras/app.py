import streamlit as st
import cv2
import numpy as np
import joblib
import pandas as pd
from PIL import Image
from datetime import datetime
import os
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Rice Quality - Ipsala",
    page_icon="🌾",
    layout="wide"
)

# Inisialisasi Session State untuk Riwayat
if 'history' not in st.session_state:
    st.session_state['history'] = []

# Fungsi Load Model & Scaler (Cached)
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
    except Exception as e:
        return None, None, None

model, scaler, le = load_resources()

# Fungsi Ekstraksi Fitur dari Kontur
def extract_feature_from_contour(cnt, img_gray, img_rgb):
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    x, y, w, h = cv2.boundingRect(cnt)
    
    aspect_ratio = float(w)/h if h != 0 else 0
    roundness = (4 * np.pi * area) / (perimeter**2) if perimeter != 0 else 0
    
    mask = np.zeros(img_gray.shape, np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    mean_val = cv2.mean(img_rgb, mask=mask)
    
    # Fitur Raw (Belum diskalakan)
    features = [area, perimeter, aspect_ratio, roundness, mean_val[0], mean_val[1], mean_val[2]]
    return features, (x, y, w, h)

# Sidebar Navigasi
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

# Halaman Aplikasi Utama
if menu == "Aplikasi Utama":
    st.title("🌾 Klasifikasi Mutu Beras (Ipsala)")
    st.write("Sistem otomatis mengkalibrasi ukuran (Auto-Scaling).")

    if model is None:
        st.error("⚠️ File Model (.pkl) tidak ditemukan. Pastikan file model, scaler, dan label encoder ada di folder yang sama.")
        st.stop()

    uploaded_file = st.file_uploader("Pilih Gambar", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        col_img, col_res = st.columns(2)
        image = Image.open(uploaded_file)
        base_width = 1000
        w_percent = (base_width / float(image.size[0]))
        h_size = int((float(image.size[1]) * float(w_percent)))
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
        
        # --- TAHAP 1: SCANNING GLOBAL (ADAPTIVE SCALING) ---
        valid_areas = []
        temp_features = [] 
        
        AVG_AREA_TRAINING = 12000 
        
        max_area_in_image = 0
        if contours:
            max_area_in_image = max([cv2.contourArea(c) for c in contours])
            
        dynamic_threshold = max(100, max_area_in_image * 0.01)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > dynamic_threshold:
                features_raw, bbox = extract_feature_from_contour(cnt, img_gray, img_array)
                valid_areas.append(area)
                temp_features.append({'cnt': cnt, 'feat': features_raw, 'bbox': bbox})
        
        scale_factor = 1.0
        if valid_areas:
            avg_area_current = np.mean(valid_areas)
            scale_factor = AVG_AREA_TRAINING / avg_area_current
        
        # --- TAHAP 2: KALIBRASI & PREDIKSI ---
        results_label = []
        results_data = []
        img_result = img_array.copy()
        count_obj = 0
        
        for item in temp_features:
            count_obj += 1
            cnt = item['cnt']
            feat_raw = item['feat']
            (x, y, w, h) = item['bbox']
            
            # Kalibrasi Area & Perimeter
            area_corrected = feat_raw[0] * scale_factor
            perimeter_corrected = feat_raw[1] * np.sqrt(scale_factor)
            
            features_corrected = [
                area_corrected, 
                perimeter_corrected, 
                feat_raw[2], 
                feat_raw[3], 
                feat_raw[4], feat_raw[5], feat_raw[6] 
            ]
            
            cols = ['Area', 'Perimeter', 'AspectRatio', 'Roundness', 'Red_Mean', 'Green_Mean', 'Blue_Mean']
            features_df = pd.DataFrame([features_corrected], columns=cols)
            features_scaled = scaler.transform(features_df)
            
            prediction_idx = model.predict(features_scaled)[0]
            label_name = le.inverse_transform([prediction_idx])[0]
            
            # LOGIKA PENYELAMAT (ANTI-ZOOM RESCUE)
            rect = cv2.minAreaRect(cnt)
            (center), (dim1, dim2), angle = rect
            long_side = max(dim1, dim2)
            short_side = min(dim1, dim2)
            real_aspect_ratio = long_side / short_side if short_side > 0 else 0
            
            if label_name == 'Broken' and real_aspect_ratio > 2.0:
                label_name = 'Whole'
            elif real_aspect_ratio < 1.5:
                label_name = 'Broken'
            
            results_label.append(label_name)
            
            row = feat_raw.copy() 
            row.insert(0, count_obj)
            row.append(label_name)
            results_data.append(row)
            
            # Visualisasi
            color_map = {'Whole': (0, 255, 0), 'Broken': (255, 0, 0), 'Chalky': (255, 0, 255), 'Discolored': (255, 255, 0)}
            color = color_map.get(label_name, (255, 255, 255))
            
            cv2.rectangle(img_result, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img_result, label_name, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        with col_res:
            st.subheader("🔍 Hasil Deteksi")
            st.image(img_result, use_container_width=True, caption=f"Total: {count_obj} | Scale Factor: {scale_factor:.2f}x")
            
            if scale_factor > 1.5:
                st.info(f"ℹ️ **Info:** Kamera terdeteksi jauh. Sistem memperbesar data fitur sebesar **{scale_factor:.1f}x** agar sesuai standar model.")

        # Menampilkan Statistik
        if results_label:
            st.markdown("---")
            st.subheader("📊 Analisis Mutu")

            df_count = pd.DataFrame(results_label, columns=['Mutu'])
            counts = df_count['Mutu'].value_counts()
            
            total = len(results_label)
            whole_pct = (counts.get('Whole', 0) / total) * 100 if total > 0 else 0
            
            status = "Sangat Baik (Premium)" if whole_pct > 80 else "Cukup Baik (Medium)" if whole_pct > 50 else "Kurang Baik (Rendah)"
            alert_type = st.success if whole_pct > 80 else st.warning if whole_pct > 50 else st.error
            alert_type(f"**Kesimpulan:** Sampel beras tergolong **{status}** dengan persentase butir utuh sebesar **{whole_pct:.1f}%**.")

            c1, c2 = st.columns([1, 2])
            with c1:
                st.write("**Distribusi Kelas:**")
                # Diagram Batang Distribusi Kelas
                st.bar_chart(counts)
            
            with c2:
                st.write("**Detail Fitur per Butir:**")
                df_table = pd.DataFrame(results_data, columns=['ID', 'Area', 'Perimeter', 'Aspect', 'Roundness', 'R', 'G', 'B', 'Kelas'])
                num_cols = ['Area', 'Perimeter', 'Aspect', 'Roundness', 'R', 'G', 'B']
                df_table[num_cols] = df_table[num_cols].apply(pd.to_numeric, errors='coerce').round(2)
                st.dataframe(df_table, use_container_width=True, hide_index=True)
                
                csv = df_table.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Laporan CSV", csv, f"laporan_mutu_beras_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")

            current_log = {
                "Waktu": datetime.now().strftime("%H:%M:%S"),
                "File": uploaded_file.name,
                "Total": count_obj,
                "Whole": counts.get('Whole', 0),
                "Broken": counts.get('Broken', 0),
                "Chalky": counts.get('Chalky', 0),
                "Discolored": counts.get('Discolored', 0),
                "Status": status
            }
            if not st.session_state['history'] or st.session_state['history'][-1]['File'] != uploaded_file.name:
                st.session_state['history'].append(current_log)

        else:
            st.warning("Objek tidak terdeteksi. Coba gunakan gambar dengan kontras lebih tinggi.")

    st.markdown("---")
    with st.expander("📈 Insight Model (Feature Importance)"):
        col_a, col_b = st.columns([1, 1])
        with col_a:
            # MEMBUAT PLOT FEATURE IMPORTANCE SECARA DINAMIS
            try:
                # Ambil feature importance dari model
                importances = model.feature_importances_
                feature_names = ['Area', 'Perimeter', 'AspectRatio', 'Roundness', 'Red_Mean', 'Green_Mean', 'Blue_Mean']
                
                # Buat DataFrame
                fi_df = pd.DataFrame({'Fitur': feature_names, 'Importance': importances})
                fi_df = fi_df.sort_values(by='Importance', ascending=False)
                
                # Plot menggunakan Matplotlib & Seaborn
                fig, ax = plt.subplots(figsize=(5, 6))
                sns.barplot(x='Importance', y='Fitur', data=fi_df, palette='viridis', ax=ax)
                ax.set_title('Feature Importance (XGBoost)')
                ax.set_xlabel('Score')
                ax.set_ylabel('Fitur')
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Gagal membuat grafik: {e}")
                
        with col_b:
            st.markdown("""
            **Interpretasi Grafik:**
            Grafik di samping dibuat otomatis dari model XGBoost yang Anda load.
            
            1.  **Fitur Morfologi (Area/Perimeter/Aspect Ratio):**
                * Sangat dominan untuk membedakan kelas **Whole** (Utuh) dan **Broken** (Patah).
            
            2.  **Fitur Warna (Mean RGB):**
                * Dominan untuk membedakan kelas **Chalky** dan **Discolored**.
            """)

elif menu == "Riwayat Analisis":
    st.title("📜 Log Riwayat Analisis")
    if st.session_state['history']:
        df_hist = pd.DataFrame(st.session_state['history'])
        st.dataframe(df_hist, use_container_width=True)
        if st.button("Hapus Semua Riwayat"):
            st.session_state['history'] = []
            st.rerun()
    else:
        st.info("Belum ada data analisis tersimpan.")