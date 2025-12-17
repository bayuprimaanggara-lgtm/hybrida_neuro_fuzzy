import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# --- 1. LOAD ASSETS ---
@st.cache_resource
def load_assets():
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

# --- 2. INTERFACE ---
st.title("🌙 Hybrid AI Sleep Disorder Detector")

with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", le_gender.classes_)
        age = st.number_input("Age", 10, 100, 30)
        occupation = st.selectbox("Occupation", le_occupation.classes_)
        sleep_duration = st.slider("Sleep Duration", 4.0, 10.0, 7.0)
        physical_activity = st.slider("Physical Activity", 0, 120, 30)
    with col2:
        stress_level = st.slider("Stress Level", 1, 10, 5)
        bmi_category = st.selectbox("BMI Category", le_bmi.classes_)
        heart_rate = st.number_input("Heart Rate", 60, 100, 72)
        daily_steps = st.number_input("Daily Steps", 0, 20000, 5000)
        # Input Tekanan Darah
        c1, c2 = st.columns(2)
        systolic = c1.number_input("Systolic", 80, 200, 120)
        diastolic = c2.number_input("Diastolic", 50, 150, 80)
    
    submit = st.form_submit_button("Prediksi")

# --- 3. LOGIKA PREDIKSI (Sesuai xf_train: 13 Fitur) ---
if submit:
    try:
        # A. Pre-scale 8 kolom numerik pertama
        # num_cols: Age, Sleep Duration, Stress Level, Heart Rate, Daily Steps, Activity, BP_Sys, BP_Dia
        num_features = np.array([[
            age, sleep_duration, stress_level, heart_rate, 
            daily_steps, physical_activity, systolic, diastolic
        ]])
        
        num_scaled = scaler.transform(num_features)
        
        # B. Ambil nilai yang sudah di-scale untuk Fuzzy Logic
        # Sesuai urutan num_cols: index 1 (Sleep Duration), index 2 (Stress Level)
        scaled_sleep = num_scaled[0, 1]
        scaled_stress = num_scaled[0, 2]
        
        # HITUNG FUZZY FEATURES (Logika dari notebook kamu)
        fuzzy_short_sleep = max(0, (0.5 - scaled_sleep))
        fuzzy_high_stress = max(0, (scaled_stress - 0.5))

        # C. Encoding Kategori
        gender_enc = le_gender.transform([gender])[0]
        bmi_enc = le_bmi.transform([bmi_category])[0]
        occ_enc = le_occupation.transform([occupation])[0]

        # D. GABUNGKAN SEMUANYA JADI 13 FITUR
        # Urutan: 8 numerik_scaled + 3 kategori_encoded + 2 fuzzy
        final_input = np.concatenate([
            num_scaled, 
            np.array([[gender_enc, bmi_enc, occ_enc, fuzzy_short_sleep, fuzzy_high_stress]])
        ], axis=1)

        # E. PREDIKSI
        prediction = dnf_model.predict(final_input)
        pred_class = np.argmax(prediction)
        result = le_target.inverse_transform([pred_class])[0]
        
        st.subheader(f"Hasil: {result}")
        st.write(f"Confidence: {np.max(prediction)*100:.2f}%")

    except Exception as e:
        st.error(f"Error: {e}")
