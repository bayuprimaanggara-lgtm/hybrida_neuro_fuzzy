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

# Custom CSS untuk mempercantik tampilan
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
    }
    </style>
    """, unsafe_allow_stdio=True)

# --- 2. LOAD ASSETS (MODEL & PREPROCESSOR) ---
@st.cache_resource
def load_assets():
    # Pastikan nama file ini sesuai dengan yang di-upload ke GitHub
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
    st.error(f"Gagal memuat assets: {e}")
    st.stop()

# --- 3. INTERFACE PENGGUNA ---
st.title("🌙 Hybrid AI: Sleep Disorder Detector")
st.write("Aplikasi cerdas berbasis Deep Learning untuk mendeteksi kesehatan tidur Anda.")
st.info("Silakan isi data di bawah ini dengan jujur untuk hasil yang akurat.")

with st.form("input_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Profil & Kebiasaan")
        gender = st.selectbox("Jenis Kelamin", le_gender.classes_)
        age = st.number_input("Usia", 10, 100, 30)
        occupation = st.selectbox("Pekerjaan", le_occupation.classes_)
        sleep_duration = st.slider("Durasi Tidur (Jam)", 4.0, 10.0, 7.0)
        quality_sleep = st.slider("Kualitas Tidur (1-10)", 1, 10, 7)
        physical_activity = st.slider("Aktivitas Fisik (Menit/Hari)", 0, 120, 30)

    with col2:
        st.subheader("Data Medis")
        stress_level = st.slider("Tingkat Stres (1-10)", 1, 10, 5)
        bmi_category = st.selectbox("Kategori BMI", le_bmi.classes_)
        heart_rate = st.number_input("Denyut Jantung (bpm)", 60, 100, 72)
        daily_steps = st.number_input("Langkah Harian", 0, 20000, 5000)
        
        st.write("**Tekanan Darah (BP)**")
        c1, c2 = st.columns(2)
        systolic = c1.number_input("Sistolik", 80, 200, 120)
        diastolic = c2.number_input("Diastolik", 50, 150, 80)
    
    submit = st.form_submit_button("Mulai Analisis AI")

# --- 4. LOGIKA PREDIKSI ---
if submit:
    try:
        # A. Pre-scale 8 kolom numerik (Sesuai fit scaler di notebook)
        num_features = np.array([[
            age, sleep_duration, stress_level, heart_rate, 
            daily_steps, physical_activity, systolic, diastolic
        ]])
        
        num_scaled = scaler.transform(num_features)
        
        # B. Hitung Fuzzy Features (Hybrid Logic)
        scaled_sleep = num_scaled[0, 1]
        scaled_stress = num_scaled[0, 2]
        fuzzy_short_sleep = max(0, (0.5 - scaled_sleep))
        fuzzy_high_stress = max(0, (scaled_stress - 0.5))

        # C. Encoding Kategori
        gender_enc = le_gender.transform([gender])[0]
        bmi_enc = le_bmi.transform([bmi_category])[0]
        occ_enc = le_occupation.transform([occupation])[0]

        # D. Gabungkan jadi 13 Fitur (Urutan xf_train)
        # Urutan: 8 numerik_scaled + 3 kategori_encoded + 2 fuzzy
        final_input = np.concatenate([
            num_scaled, 
            np.array([[gender_enc, bmi_enc, occ_enc, fuzzy_short_sleep, fuzzy_high_stress]])
        ], axis=1)

        # E. Eksekusi Model
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
                st.success(f"### Kondisi Anda: **SEHAT** ✨")
                st.write("Luar biasa! Tubuh Anda menunjukkan pola tidur yang baik. Pertahankan!")
                st.balloons()
            elif result_label == "Insomnia":
                st.warning(f"### Kondisi Anda: **INSOMNIA** ⚠️")
                st.write("Ada indikasi kesulitan memulai atau mempertahankan tidur.")
            elif result_label == "Sleep Apnea":
                st.error(f"### Kondisi Anda: **SLEEP APNEA** 🚨")
                st.write("Terdeteksi pola gangguan pernapasan saat tidur.")

        with res_col2:
            st.metric(label="AI Confidence", value=f"{confidence:.1f}%")

        # SEKSI SARAN
        st.write("---")
        st.write("### 💡 Saran & Penyemangat:")
        
        if result_label == "Insomnia":
            st.markdown("""
            * **Power Down:** Matikan gadget 1 jam sebelum tidur. Cahaya biru itu musuhmu!
            * **Mindfulness:** Coba meditasi atau teknik pernapasan 4-7-8 sebelum tidur.
            * *Jangan menyerah, pikiran yang tenang adalah kunci tidur yang nyenyak. Kamu pasti bisa melewati ini!*
            """)
        elif result_label == "Sleep Apnea":
            st.markdown("""
            * **Posisi Tidur:** Cobalah tidur menyamping agar jalan napas lebih terbuka.
            * **Konsultasi Dokter:** Sangat disarankan untuk cek ke dokter spesialis tidur (Sleep Clinic).
            * *Kesehatanmu berharga. Mengambil langkah untuk berobat adalah bukti kamu sayang dirimu sendiri!*
            """)
        else:
            st.markdown("""
            * **Consistency:** Tetap bangun dan tidur di jam yang sama setiap hari.
            * **Environment:** Pastikan kamar tidurmu gelap, sejuk, dan tenang.
            * *Tubuh yang bugar berawal dari tidur yang benar. Teruslah menjadi inspirasi sehat!*
            """)
            
        st.caption("Catatan: Ini adalah hasil prediksi model AI. Konsultasikan dengan tenaga medis untuk diagnosa resmi.")

    except Exception as e:
        st.error(f"Terjadi kesalahan teknis: {e}")
