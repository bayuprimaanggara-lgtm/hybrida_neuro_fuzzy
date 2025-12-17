import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Sleep Disorder Detector", page_icon="🌙")

# --- LOAD MODEL & PREPROCESSOR ---
@st.cache_resource
def load_assets():
    # Pastikan nama file ini sama dengan yang Anda simpan di notebook
    model = tf.keras.models.load_model('model_sleep.h5')
    scaler = joblib.load('scaler.pkl')
    le_gender = joblib.load('le_gender.pkl')
    le_occupation = joblib.load('le_occ.pkl')
    le_bmi = joblib.load('le_bmi.pkl')
    le_target = joblib.load('le_target.pkl') # Encoder untuk Healthy, Insomnia, dll
    return model, scaler, le_gender, le_occupation, le_bmi, le_target

try:
    dnf_model, scaler, le_gender, le_occupation, le_bmi, le_target = load_assets()
except Exception as e:
    st.error(f"Error loading assets: {e}")
    st.stop()

# --- INTERFACE ---
st.title("🌙 Sleep Disorder Classification")
st.write("Masukkan data kesehatan Anda untuk memprediksi potensi gangguan tidur.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", le_gender.classes_)
        age = st.number_input("Age", min_value=10, max_value=100, value=30)
        occupation = st.selectbox("Occupation", le_occupation.classes_)
        sleep_duration = st.slider("Sleep Duration (Hours)", 4.0, 10.0, 7.0)
        quality_sleep = st.slider("Quality of Sleep (1-10)", 1, 10, 7)
        physical_activity = st.slider("Physical Activity Level (min/day)", 0, 120, 30)

    with col2:
        stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)
        bmi_category = st.selectbox("BMI Category", le_bmi.classes_)
        # Input Tekanan Darah manual
        bp_input = st.text_input("Blood Pressure (Systolic/Diastolic)", "120/80")
        heart_rate = st.number_input("Heart Rate (bpm)", 60, 100, 72)
        daily_steps = st.number_input("Daily Steps", 0, 15000, 5000)

    submit = st.form_submit_button("Cek Hasil Prediksi")

# --- PROSES PREDIKSI ---
# --- DI DALAM app.py ---
if submit:
    try:
        # 1. Split Blood Pressure
        systolic, diastolic = map(int, bp_input.split('/'))
        
        # 2. Transform Categorical (Encoding)
        gender_enc = le_gender.transform([gender])[0]
        occ_enc = le_occupation.transform([occupation])[0]
        bmi_enc = le_bmi.transform([bmi_category])[0]

        # 3. SUSUN ARRAY SESUAI X_TRAIN ANDA (PENTING!)
        # Urutan: Age, Sleep Duration, Stress Level, Heart Rate, Daily Steps, 
        # Physical Activity Level, BP_Sys, BP_Dia, Gender_Code, BMI_Code, Occ_Code
        features = np.array([[
            age,                # 1
            sleep_duration,     # 2
            stress_level,       # 3
            heart_rate,         # 4
            daily_steps,        # 5
            physical_activity,  # 6
            systolic,           # 7 (BP_Sys)
            diastolic,          # 8 (BP_Dia)
            gender_enc,         # 9 (Gender_Code)
            bmi_enc,            # 10 (BMI_Code)
            occ_enc             # 11 (Occ_Code)
        ]])

        # 4. Scaling (Hanya kolom numerik yang discale jika scaler Anda hanya fit pada num_cols)
        # TAPI, biasanya scaler di-fit ke seluruh X_train. 
        # Jika scaler Anda di-fit ke 'X' (11 kolom), maka:
        features_scaled = scaler.transform(features)

        # 5. Predict
        prediction = dnf_model.predict(features_scaled)
        pred_class = np.argmax(prediction)
        result_label = le_target.inverse_transform([pred_class])[0]
        confidence = np.max(prediction) * 100

        # 6. Tampilkan Hasil
        st.subheader("Hasil Analisis:")
        if result_label == "Healthy":
            st.success(f"Kondisi: **{result_label}**")
        else:
            st.warning(f"Terdeteksi Potensi: **{result_label}**")
        
        st.write(f"Tingkat Keyakinan Model: {confidence:.2f}%")
        
    except ValueError:
        st.error("Format Blood Pressure salah! Gunakan format '120/80'")
    except Exception as e:
        st.error(f"Terjadi kesalahan teknis: {e}")