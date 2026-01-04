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
        
    except Exception as e:
        st.error(f"🔥 Error Load File: {e}")
        st.write(f"📂 Folder Script: {os.path.dirname(os.path.abspath(__file__))}")
        st.write(f"📄 Daftar File di sini: {os.listdir(os.path.dirname(os.path.abspath(__file__)))}")
        return None, None, None

model, scaler, le = load_resources()

def extract_feature_from_contour(cnt, img_gray, img_rgb):
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    x, y, w, h = cv2.boundingRect(cnt) 
    
    aspect_ratio = float(w)/h if h != 0 else 0
    roundness = (4 * np.pi * area) / (perimeter**2) if perimeter != 0 else 0
    
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
    st.write("**Kelas:** Whole, Broken, Chalky, Discolored")

st.sidebar.markdown("---")
st.sidebar.info("© 2026 - Data Science")

if menu == "Aplikasi Utama":
    st.title("🌾 Klasifikasi Mutu Beras (Ipsala)")
    st.write("Upload citra beras dengan latar belakang gelap.")

    if model is None:
        st.error("⚠️ File Model (.pkl) tidak ditemukan.")
        st.stop()

    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        col_img, col_res = st.columns(2)
        
        image = Image.open(uploaded_file)
        
        base_width = 1000
        w_percent = (base_width / float(image.size[0]))
        h_size = int((float(image.size[1]) * float(w_percent)))
        image = image.resize((base_width, h_size), Image.Resampling.LANCZOS)

        with col_img:
            st.subheader("📸 Citra Input")
            st.image(image, use_container_width=True)

        img_array = np.array(image.convert('RGB'))
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        results_label = []
        results_data = []
        img_result = img_array.copy()
        count_obj = 0
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            if area > 100: 
                count_obj += 1
                features_raw, (x, y, w, h) = extract_feature_from_contour(cnt, img_gray, img_array)
                
                cols = ['Area', 'Perimeter', 'AspectRatio', 'Roundness', 'Red_Mean', 'Green_Mean', 'Blue_Mean']
                features_df = pd.DataFrame([features_raw], columns=cols)
                features_scaled = scaler.transform(features_df)
                
                prediction_idx = model.predict(features_scaled)[0]
                label_name = le.inverse_transform([prediction_idx])[0]
                
                rect = cv2.minAreaRect(cnt)
                (center), (dim1, dim2), angle = rect
                long_side = max(dim1, dim2)
                short_side = min(dim1, dim2)
                
                real_aspect_ratio = long_side / short_side if short_side > 0 else 0
                
                if real_aspect_ratio < 1.8:
                    label_name = 'Broken'
                
                results_label.append(label_name)
                
                row = features_raw.copy()
                row.insert(0, count_obj)
                row.append(label_name)
                results_data.append(row)
                
                color_map = {'Whole': (0, 255, 0), 'Broken': (255, 0, 0), 'Chalky': (255, 0, 255), 'Discolored': (255, 255, 0)}
                color = color_map.get(label_name, (255, 255, 255))
                
                cv2.rectangle(img_result, (x, y), (x+w, y+h), color, 2)
                cv2.putText(img_result, label_name, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        with col_res:
            st.subheader("🔍 Hasil Deteksi")
            st.image(img_result, use_container_width=True, caption=f"Total Objek: {count_obj}")

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
                st.bar_chart(counts)
            
            with c2:
                st.write("**Detail Fitur:**")
                df_table = pd.DataFrame(results_data, columns=['ID', 'Area', 'Perimeter', 'Aspect', 'Roundness', 'R', 'G', 'B', 'Kelas'])
                
                num_cols = ['Area', 'Perimeter', 'Aspect', 'Roundness', 'R', 'G', 'B']
                df_table[num_cols] = df_table[num_cols].apply(pd.to_numeric, errors='coerce').round(2)
                
                st.dataframe(df_table, use_container_width=True, hide_index=True)
                
                csv = df_table.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download CSV", csv, f"laporan_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")

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
            st.warning("Objek tidak terdeteksi.")

    st.markdown("---")
    with st.expander("📈 Insight Model (Feature Importance)"):
        try:
            st.image("feature_importance.png", use_container_width=True)
        except:
            st.caption("Grafik feature_importance.png tidak ditemukan.")

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