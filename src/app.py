import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Başlık ve açıklama
st.title("Kalp Hastalığı Tahmini Uygulaması")
st.write("Bu uygulama, girilen sağlık verilerine göre kalp hastalığı riskini tahmin eder.")

# Model ve scaler'ı yükle
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

try:
    model, scaler = load_model()
    st.success("Model başarıyla yüklendi!")
except Exception as e:
    st.error(f"Model yüklenirken hata oluştu: {e}")
    st.info("Lütfen önce 'train_model.py' dosyasını çalıştırarak modeli eğitin.")
    st.stop()

# Kullanıcı girdileri
st.subheader("Hasta Bilgileri")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Yaş", min_value=20, max_value=100, value=40)
    sex = st.selectbox("Cinsiyet", options=["Erkek", "Kadın"])
    chest_pain = st.selectbox("Göğüs Ağrısı Tipi", 
                             options=["ATA (Tipik Angina)", 
                                      "NAP (Atipik Angina)", 
                                      "ASY (Asemptomatik)", 
                                      "TA (Tipik Olmayan Ağrı)"])
    resting_bp = st.number_input("İstirahat Kan Basıncı (mmHg)", min_value=80, max_value=200, value=120)
    cholesterol = st.number_input("Kolesterol (mg/dl)", min_value=100, max_value=600, value=200)
    fasting_bs = st.selectbox("Açlık Kan Şekeri > 120 mg/dl", options=["Hayır", "Evet"])

with col2:
    resting_ecg = st.selectbox("Dinlenme EKG Sonucu", 
                              options=["Normal", "ST (ST-T Dalgası Anormalliği)", "LVH (Sol Ventrikül Hipertrofisi)"])
    max_hr = st.number_input("Maksimum Kalp Atış Hızı", min_value=60, max_value=220, value=150)
    exercise_angina = st.selectbox("Egzersize Bağlı Anjina", options=["Hayır", "Evet"])
    oldpeak = st.number_input("ST Segmentindeki Düşüş", min_value=0.0, max_value=10.0, value=0.0, format="%.1f")
    st_slope = st.selectbox("ST Eğimi", options=["Up (Yukarı)", "Flat (Düz)", "Down (Aşağı)"])

# Veri dönüşümleri
def preprocess_input():
    # Kategorik değişkenleri sayısal değerlere dönüştür
    sex_encoded = 1 if sex == "Erkek" else 0
    
    chest_pain_mapping = {
        "ATA (Tipik Angina)": "ATA", 
        "NAP (Atipik Angina)": "NAP", 
        "ASY (Asemptomatik)": "ASY", 
        "TA (Tipik Olmayan Ağrı)": "TA"
    }
    chest_pain_encoded = chest_pain_mapping[chest_pain]
    
    fasting_bs_encoded = 1 if fasting_bs == "Evet" else 0
    
    resting_ecg_mapping = {
        "Normal": "Normal", 
        "ST (ST-T Dalgası Anormalliği)": "ST", 
        "LVH (Sol Ventrikül Hipertrofisi)": "LVH"
    }
    resting_ecg_encoded = resting_ecg_mapping[resting_ecg]
    
    exercise_angina_encoded = "Y" if exercise_angina == "Evet" else "N"
    
    st_slope_mapping = {
        "Up (Yukarı)": "Up", 
        "Flat (Düz)": "Flat", 
        "Down (Aşağı)": "Down"
    }
    st_slope_encoded = st_slope_mapping[st_slope]
    
    # DataFrame oluştur
    input_df = pd.DataFrame({
        'Age': [age],
        'Sex': [sex_encoded],
        'ChestPainType': [chest_pain_encoded],
        'RestingBP': [resting_bp],
        'Cholesterol': [cholesterol],
        'FastingBS': [fasting_bs_encoded],
        'RestingECG': [resting_ecg_encoded],
        'MaxHR': [max_hr],
        'ExerciseAngina': [exercise_angina_encoded],
        'Oldpeak': [oldpeak],
        'ST_Slope': [st_slope_encoded]
    })
    
    # Kategorik değişkenleri sayısal hale getir (eğitim kodundaki ile aynı şekilde)
    input_df['Sex'] = input_df['Sex'].replace({'M': 1, 'F': 0, 1: 1, 0: 0})
    input_df['ChestPainType'] = input_df['ChestPainType'].replace({'ATA': 0, 'NAP': 1, 'ASY': 2, 'TA': 3})
    input_df['RestingECG'] = input_df['RestingECG'].replace({'Normal': 0, 'ST': 1, 'LVH': 2})
    input_df['ExerciseAngina'] = input_df['ExerciseAngina'].replace({'N': 0, 'Y': 1})
    input_df['ST_Slope'] = input_df['ST_Slope'].replace({'Up': 0, 'Flat': 1, 'Down': 2})
    
    return input_df

# Tahmin butonu
if st.button("Tahmin Et", type="primary"):
    # Kullanıcı girdilerini işle
    input_df = preprocess_input()
    
    # Veriyi ölçeklendir
    input_scaled = scaler.transform(input_df)
    
    # Tahmin yap
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)
    
    # Sonuçları göster
    st.subheader("Tahmin Sonucu")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if prediction[0] == 1:
            st.error("⚠️ Kalp Hastalığı Riski VAR")
        else:
            st.success("✅ Kalp Hastalığı Riski YOK")
    
    with col2:
        risk_percentage = round(prediction_proba[0][1] * 100, 2)
        st.metric("Risk Yüzdesi", f"%{risk_percentage}")
    
    # Risk faktörleri açıklaması
    st.subheader("Risk Faktörleri Analizi")
    
    # Yaş
    if age > 55:
        st.warning("Yaş faktörü: 55 yaş üstü kişilerde kalp hastalığı riski daha yüksektir.")
    
    # Kolesterol
    if cholesterol > 240:
        st.warning("Kolesterol faktörü: 240 mg/dl üzerindeki kolesterol değerleri risk faktörüdür.")
    
    # Kan basıncı
    if resting_bp > 140:
        st.warning("Kan basıncı faktörü: 140 mmHg üzerindeki kan basıncı değerleri risk faktörüdür.")
    
    # Maksimum kalp hızı
    if max_hr < 100:
        st.warning("Kalp hızı faktörü: Düşük maksimum kalp hızı risk faktörü olabilir.")
    
    # ST segmenti
    if oldpeak > 2:
        st.warning("ST segmenti faktörü: 2'den büyük ST segmenti düşüşü önemli bir risk faktörüdür.")
