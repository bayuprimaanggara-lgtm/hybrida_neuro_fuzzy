import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Sleep Disorder Detector", page_icon="🌙")

# --- 2. LOAD MODEL & ASSETS ---
@st.cache_resource
def load_assets():
    # Pastikan file ini sudah di-upload ke GitHub
    model = tf.keras.models.load_model('model_sleep.h5')
    scaler = joblib.load('scaler.pkl')
    le_gender = joblib.load('le_gender.pkl')
    le_occupation = joblib.load('le_occ.pkl') # Sesuai nama di kode terakhirmu
    le_bmi = joblib.load('le_bmi.pkl')
    le_target = joblib.load('le_target.pkl')
    return model, scaler, le_gender, le_occupation, le_bmi, le_target

try:
    dnf_model, scaler, le_gender, le_occupation, le_bmi, le_target = load_assets()
except Exception as e:
    st.error(f"Gagal memuat file pendukung: {e}")
    st.stop()

# --- 3. INTERFACE ---
st.title("🌙 Sleep Disorder Classification")
st.write("Gunakan formulir di bawah untuk mendeteksi potensi gangguan tidur.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Jenis Kelamin", le_gender.classes_)
        age = st.number_input("Usia", min_value=10, max_value=100, value=30)
        occupation = st.selectbox("Pekerjaan", le_occupation.classes_)
        sleep_duration = st.slider("Durasi Tidur (Jam)", 4.0, 10.0, 7.0)
        physical_activity = st.slider("Aktivitas Fisik (Menit/Hari)", 0, 120, 30)

    with col2:
        stress_level = st.slider("Tingkat Stres (1-10)", 1, 10, 5)
        bmi_category = st.selectbox("Kategori BMI", le_bmi.classes_)
        
        # INPUT TEKANAN DARAH TERPISAH (Agar tidak ada lagi error format)
        st.write("**Tekanan Darah (BP)**")
        c1, c2 = st.columns(2)
        systolic = c1.number_input("Sistolik", min_value=80, max_value=200, value=120)
        diastolic = c2.number_input("Diastolik", min_value=50, max_value=150, value=80)
        
        heart_rate = st.number_input("Denyut Jantung (bpm)", 60, 100, 72)
        daily_steps = st.number_input("Langkah Harian", 0, 20000, 5000)

    submit = st.form_submit_button("Cek Hasil Prediksi")

# --- 4. PROSES PREDIKSI ---
# --- 4. PROSES PREDIKSI (VERSI FIX SCALER) ---
if submit:
    try:
        # A. Encoding Kategori (Kategori tidak ikut di-scale)
        gender_enc = le_gender.transform([gender])[0]
        occ_enc = le_occupation.transform([occupation])[0]
        bmi_enc = le_bmi.transform([bmi_category])[0]

        # B. Pisahkan Kolom Numerik (Sesuai urutan num_cols di notebook)
        # num_cols = ['Age','Sleep Duration','Stress Level','Heart Rate','Daily Steps','Physical Activity Level','BP_Sys','BP_Dia']
        num_features = np.array([[
            age,                # 1
            sleep_duration,     # 2
            stress_level,       # 3
            heart_rate,         # 4
            daily_steps,        # 5
            physical_activity,  # 6
            systolic,           # 7
            diastolic           # 8
        ]])

        # C. Scaling hanya untuk 8 kolom numerik
        num_features_scaled = scaler.transform(num_features)

        # D. Gabungkan 8 kolom hasil scaling + 3 kolom kategori (Total 11)
        # Urutan harus: num_cols + ['Gender_Code','BMI_Code','Occ_Code']
        final_features = np.concatenate([
            num_features_scaled, 
            np.array([[gender_enc, bmi_enc, occ_enc]])
        ], axis=1)

        # E. Predict (Sekarang inputnya sudah pas 11 fitur)
        prediction = dnf_model.predict(final_features)
        pred_class = np.argmax(prediction)
        
        # F. Decode Hasil
        result_label = le_target.inverse_transform([pred_class])[0]
        confidence = np.max(prediction) * 100

        # Tampilkan Hasil
        st.divider()
        if result_label.lower() in ["healthy", "none", "normal"]:
            st.success(f"Hasil: **{result_label}**")
        else:
            st.warning(f"Hasil: **{result_label}**")
        st.write(f"Confidence: {confidence:.2f}%")
        
    except Exception as e:
        st.error(f"Terjadi kesalahan teknis: {e}")
