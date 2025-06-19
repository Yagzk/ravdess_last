import os
import librosa
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ================================
# Yardımcı fonksiyonlar
# ================================
def aug_extract_features(y, sr):
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y).T, axis=0)
    rms = np.mean(librosa.feature.rms(y=y).T, axis=0)
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).T, axis=0)
    return np.hstack([mfccs, chroma, contrast, zcr, rms, centroid])

# ================================
# Streamlit Uygulaması
# ================================
st.set_page_config(page_title="🎤 Ses Duygu Tanıma", layout="wide")
st.title("🎵 Ses Duygu Tanıma ve Sınıflandırma (Hızlı Tahmin)")
st.caption("Eğitim yapılmadan, doğrudan önceden eğitilmiş modelle çalışır.")

with st.spinner("🔄 Model yükleniyor..."):
    try:
        AUG_best_model = joblib.load("aug_best_model.pkl")
        AUG_scaler = joblib.load("aug_scaler.pkl")
        AUG_pca = joblib.load("aug_pca.pkl")
        AUG_le = joblib.load("aug_labelencoder.pkl")
        st.success("✅ Model ve yardımcı nesneler başarıyla yüklendi.")
    except Exception as e:
        st.error(f"❌ Model yüklenemedi: {e}")
        st.stop()

# Ses dosyası yükleme
st.subheader("🔊 Kendi ses dosyanı yükle ve tahmin et")
uploaded_file = st.file_uploader("Bir .wav dosyası yükleyin", type=["wav"])

if uploaded_file is not None:
    y, sr = librosa.load(uploaded_file, sr=48000)
    st.audio(uploaded_file)

    # Dalga formu
    fig_wave, ax_wave = plt.subplots(figsize=(10, 3))
    ax_wave.plot(y)
    ax_wave.set_title("Ses Dalga Formu")
    ax_wave.set_xlabel("Örnek Numarası")
    ax_wave.set_ylabel("Genlik")
    st.pyplot(fig_wave)

    # Özellik çıkarımı
    features = aug_extract_features(y, sr).reshape(1, -1)
    features_scaled = AUG_scaler.transform(features)
    features_pca = AUG_pca.transform(features_scaled)

    # Tahmin
    pred = AUG_best_model.predict(features_pca)
    pred_label = AUG_le.inverse_transform(pred)[0]
    st.success(f"🔮 **Tahmin Edilen Duygu:** {pred_label}")

    if hasattr(AUG_best_model, "predict_proba"):
        probs = AUG_best_model.predict_proba(features_pca)[0]
        fig_prob, ax_prob = plt.subplots()
        ax_prob.bar(AUG_le.classes_, probs * 100)
        ax_prob.set_ylabel("Olasılık (%)")
        ax_prob.set_title("Duygu Olasılıkları")
        st.pyplot(fig_prob)
    else:
        st.info("Bu model probability desteklemiyor.")

st.markdown("---")
st.caption("💻 *Geliştiren: Yağız | Powered by Streamlit, Librosa, Scikit-learn*")
