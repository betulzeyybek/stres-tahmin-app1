import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.set_page_config(page_title="Stres Tahmin Uygulaması", layout="centered")
st.title("🧠 Stres Tahmin Uygulaması")
st.markdown("Bu uygulama, belirli psikolojik ve davranışsal ölçütlere göre kişinin stresli olup olmadığını tahmin eder.")

# Özellik açıklamaları ve Türkçe başlıklar
feature_map = {
    'cesd': ('Depresyon Skoru (CES-D)', '70 üzeri → yüksek depresyon riski'),
    'mbi_ex': ('Duygusal Tükenmişlik (MBI-EX)', '80 üzeri → yüksek tükenmişlik'),
    'mbi_ea': ('Empati Azalması (MBI-EA)', '60 üzeri → empati kaybı olabilir'),
    'health': ('Kendi Sağlık Değerlendirmesi', '70 üzeri → sağlık algısı düşmüş olabilir'),
    'mbi_cy': ('Sorgulama/Duyarsızlaşma (MBI-CY)', '80 üzeri → duyarsızlaşma artabilir')
}

st.sidebar.header("🔧 Girdi Değerleri")
user_input = {}

for key, (label, desc) in feature_map.items():
    value = st.sidebar.slider(label, min_value=0, max_value=100, value=50)
    st.sidebar.caption(desc)
    user_input[key] = value

input_df = pd.DataFrame([user_input], columns=feature_map.keys())


with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)[0]

# Tahmin sonucu göster
st.subheader("📊 Tahmin Sonucu:")
if prediction == 1:
    st.error("🔴 Tahmin: **Stresli**")
else:
    st.success("🟢 Tahmin: **Stresli Değil**")

st.markdown("---")
st.caption("Model: KNN (Korelasyon ile seçilen 5 özellik kullanılarak eğitildi)")
