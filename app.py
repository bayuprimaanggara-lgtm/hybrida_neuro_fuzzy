import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Sleep Disorder Detector AI",
    page_icon="🌙",
    layout="centered"
)

# Custom CSS untuk mempercantik tampilan (Sudah difix untuk Python 3.13)
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 10px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOAD ASSETS ---
@st.cache_resource
def load_assets():
    # Pastikan file-file ini sudah di-upload ke GitHub Anda
    model = tf.keras.models.load_model('model_sleep.h5')
    scaler = joblib.load('scaler.pkl')
    le_gender = joblib.load('le_gender.pkl')
    le_occupation = joblib.load('le_occ.pkl') 
    le_bmi = joblib.load('le_bmi.pkl')
    le_target = joblib.load('le_target.pkl')
    return model, scaler, le_gender, le_occupation, le_bmi, le_target

try:
    dnf_model, scaler, le_gender, le_occupation, le_bmi, le_target = load_assets()
except Exception as e:
    st.error(f"⚠️ Gagal memuat model atau encoder: {e}")
    st.stop()

# --- 3. INTERFACE PENGGUNA ---
st.title("🌙 Hybrid AI: Sleep Disorder Detector")
st.write("Analisis kesehatan tidur Anda menggunakan penggabungan Deep Learning dan Fuzzy Logic.")

with st.form("input_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Data Personal")
        gender = st.selectbox("Jenis Kelamin", le_gender.classes_)
        age = st.number_input("Usia", 10, 100, 30)
        occupation = st.selectbox("Pekerjaan", le_occupation.classes_)
        sleep_duration = st.slider("Durasi Tidur (Jam)", 4.0, 10.0, 7.0)
        quality_sleep = st.slider("Kualitas Tidur (1-10)", 1, 10, 7)
        physical_activity = st.slider("Aktivitas Fisik (Menit/Hari)", 0, 120, 30)

    with col2:
        st.subheader("🏥 Parameter Medis")
        stress_level = st.slider("Tingkat Stres (1-10)", 1, 10, 5)
        bmi_category = st.selectbox("Kategori BMI", le_bmi.classes_)
        heart_rate = st.number_input("Denyut Jantung (bpm)", 60, 100, 72)
        daily_steps = st.number_input("Langkah Harian", 0, 20000, 5000)
        
        st.write("**Tekanan Darah (Systolic/Diastolic)**")
        c1, c2 = st.columns(2)
        systolic = c1.number_input("Sistolik", 80, 200, 120)
        diastolic = c2.number_input("Diastolik", 50, 150, 80)
    
    submit = st.form_submit_button("🔍 ANALISIS SEKARANG")

# --- 4. LOGIKA PREDIKSI ---
if submit:
    try:
        # A. Pre-scale 8 kolom numerik (Sesuai urutan num_cols di notebook)
        num_features = np.array([[
            age, sleep_duration, stress_level, heart_rate, 
            daily_steps, physical_activity, systolic, diastolic
        ]])
        
        num_scaled = scaler.transform(num_features)
        
        # B. Hitung Fitur Fuzzy (Logika Hybrid AI)
        scaled_sleep = num_scaled[0, 1]
        scaled_stress = num_scaled[0, 2]
        fuzzy_short_sleep = max(0, (0.5 - scaled_sleep))
        fuzzy_high_stress = max(0, (scaled_stress - 0.5))

        # C. Encoding Kategori
        gender_enc = le_gender.transform([gender])[0]
        bmi_enc = le_bmi.transform([bmi_category])[0]
        occ_enc = le_occupation.transform([occupation])[0]

        # D. Gabungkan jadi 13 Fitur (Input Model)
        # Urutan: 8 numerik_scaled + 3 kategori_encoded + 2 fuzzy
        final_input = np.concatenate([
            num_scaled, 
            np.array([[gender_enc, bmi_enc, occ_enc, fuzzy_short_sleep, fuzzy_high_stress]])
        ], axis=1)

        # E. Eksekusi Prediksi
        prediction = dnf_model.predict(final_input)
        pred_class = np.argmax(prediction)
        result_label = le_target.inverse_transform([pred_class])[0]
        confidence = np.max(prediction) * 100

        # --- 5. TAMPILAN HASIL & SARAN ---
        st.divider()
        st.subheader("📊 Hasil Analisis AI")
        
        res_col1, res_col2 = st.columns([2, 1])
        
        with res_col1:
            if result_label.lower() in ["healthy", "none", "normal"]:
                st.success(f"### Kondisi: **SEHAT (Normal)** ✨")
                st.write("Mantap! Pertahankan pola tidurmu. Tubuh yang segar adalah kunci produktivitas tinggi!")
                st.balloons()
            elif result_label == "Insomnia":
                st.warning(f"### Kondisi: **INSOMNIA** ⚠️")
                st.write("Sering sulit tidur atau terbangun di malam hari? Tenang, ini bisa diperbaiki kok.")
            elif result_label == "Sleep Apnea":
                st.error(f"### Kondisi: **SLEEP APNEA** 🚨")
                st.write("Ada indikasi gangguan pernapasan saat tidur yang perlu diperhatikan.")

        # SESI SARAN & PENYEMANGAT
        st.write("---")
        st.write("### 💡 Saran & Penyemangat Untukmu:")
        
        if result_label == "Insomnia":
            st.markdown("""
            * **Power Down:** Matikan HP/Laptop 1 jam sebelum tidur agar otak lebih rileks.
            * **Mindfulness:** Coba dengarkan musik relaksasi atau meditasi sebelum memejamkan mata.
            * *Jangan menyerah, istirahat bukan berarti kalah. Tubuhmu hanya butuh waktu untuk tenang.*
            """)
        elif result_label == "Sleep Apnea":
            st.markdown("""
            * **Posisi Tidur:** Cobalah tidur menyamping agar jalan napas tidak tertutup.
            * **Cek Medis:** Sangat disarankan untuk konsultasi ke dokter THT atau spesialis tidur.
            * *Kesehatan adalah investasi. Menangani masalah sejak dini adalah bukti kamu sayang dirimu.*
            """)
        else:
            st.markdown("""
            * **Konsistensi:** Bangun dan tidurlah di jam yang sama setiap hari agar jam biologis terjaga.
            * **Hidrasi:** Jangan lupa minum air putih yang cukup sepanjang hari.
            * *Teruslah jadi versi terbaik dirimu! Tubuh yang bugar adalah modal paling berharga.*
            """)
            
        st.caption("Catatan: Aplikasi ini hanyalah alat bantu prediksi AI. Hasilnya tidak menggantikan diagnosa dokter profesional.")

    except Exception as e:
        st.error(f"⚠️ Terjadi kesalahan teknis: {e}")

