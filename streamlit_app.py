
import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="Stres Tahmin Uygulaması", layout="centered")
st.title("🧠 Stres Tahmin Uygulaması")
st.markdown("Bu uygulama, belirli psikolojik ve davranışsal ölçütlere göre kişinin stresli olup olmadığını tahmin eder.")

features = ['cesd', 'mbi_ex', 'mbi_ea', 'health', 'mbi_cy']

user_input = []
st.sidebar.header("🔧 Girdi Değerleri")

for feature in features:
    value = st.sidebar.slider(f"{feature}", min_value=0, max_value=100, value=50)
    user_input.append(value)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

user_input_np = np.array(user_input).reshape(1, -1)
user_input_scaled = scaler.transform(user_input_np)
prediction = model.predict(user_input_scaled)[0]

st.subheader("📊 Tahmin Sonucu:")
if prediction == 1:
    st.error("🔴 Tahmin: **Stresli**")
else:
    st.success("🟢 Tahmin: **Stresli Değil**")

st.markdown("---")
st.caption("Model: KNN (Korelasyon ile seçilen 5 özellik kullanılarak eğitildi)")
