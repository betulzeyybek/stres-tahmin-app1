import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.set_page_config(page_title="Stres Tahmin Uygulaması", layout="centered")
st.title("🧠 Stres Tahmin Uygulaması")
st.markdown("Bu uygulama, belirli psikolojik ve davranışsal ölçütlere göre kişinin stresli olup olmadığını tahmin eder.")

# Teknik olarak kullanılan sütun adları (model böyle eğitildi)
columns = ['cesd', 'mbi_ex', 'mbi_ea', 'health', 'mbi_cy']

# Türkçe açıklamalar
feature_labels = {
    'cesd': ('Depresyon Skoru (CES-D)', '70 üzeri → yüksek depresyon riski'),
    'mbi_ex': ('Duygusal Tükenmişlik (MBI-EX)', '80 üzeri → yüksek tükenmişlik'),
    'mbi_ea': ('Empati Azalması (MBI-EA)', '60 üzeri → empati kaybı olabilir'),
    'health': ('Kendi Sağlık Değerlendirmesi', '70 üzeri → sağlık algısı düşmüş olabilir'),
    'mbi_cy': ('Sorgulama/Duyarsızlaşma (MBI-CY)', '80 üzeri → duyarsızlaşma artabilir')
}

st.sidebar.header("🔧 Girdi Değerleri")
values = []
for key in columns:
    label, desc = feature_labels[key]
    val = st.sidebar.slider(label, 0, 100, 50)
    st.sidebar.caption(desc)
    values.append(val)


input_df = pd.DataFrame([values], columns=columns)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)[0]


st.subheader("📊 Tahmin Sonucu:")
if prediction == 1:
    st.error("🔴 Tahmin: **Stresli**")
else:
    st.success("🟢 Tahmin: **Stresli Değil**")

st.markdown("---")
st.caption("Model: KNN (Korelasyon ile seçilen 5 özellik ile eğitildi)")
