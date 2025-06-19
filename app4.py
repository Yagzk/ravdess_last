import os
import librosa
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# ================================
# Yardımcı fonksiyonlar
# ================================
AUG_EMOTION_DICT = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

def aug_parse_filename(filename):
    parts = filename.split('.')[0].split('-')
    emotion = AUG_EMOTION_DICT[parts[2]]
    intensity = int(parts[3])
    actor = int(parts[6])
    gender = 'female' if actor % 2 == 0 else 'male'
    return emotion, intensity, gender

def aug_add_noise(y, noise_factor=0.005):
    noise = np.random.randn(len(y))
    return y + noise_factor * noise

def aug_pitch_shift(y, sr, n_steps=2):
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

def aug_time_stretch(y, rate=1.1):
    return librosa.effects.time_stretch(y, rate=rate)

def aug_extract_features(y, sr):
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y).T, axis=0)
    rms = np.mean(librosa.feature.rms(y=y).T, axis=0)
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).T, axis=0)
    return np.hstack([mfccs, chroma, contrast, zcr, rms, centroid])

def aug_extract_features_single(y, sr):
    return aug_extract_features(y, sr)

# ================================
# Streamlit Uygulaması
# ================================
st.set_page_config(page_title="🎤 Ses Duygu Tanıma", layout="wide")
st.title("🎵 Ses Duygu Tanıma ve Sınıflandırma")
st.caption("Ses dosyanızdan duygu tespiti yapan PCA + SVM modeli")

with st.spinner("🔄 Model hazırlanıyor..."):
    # Veriyi yükle veya oluştur
    if os.path.exists('dataaug.csv'):
        AUG_df_full = pd.read_csv('dataaug.csv')
    else:
        st.warning("dataaug.csv bulunamadı. Lütfen terminalde augmentasyon çalıştırın.")
        st.stop()

    AUG_X_data = AUG_df_full.drop(columns=['file', 'emotion', 'intensity', 'gender']).values
    AUG_y_data = AUG_df_full['emotion'].values

    AUG_le = LabelEncoder()
    AUG_y_encoded = AUG_le.fit_transform(AUG_y_data)

    AUG_scaler = StandardScaler()
    AUG_X_scaled = AUG_scaler.fit_transform(AUG_X_data)

    AUG_pca = PCA(n_components=0.95)
    AUG_X_pca = AUG_pca.fit_transform(AUG_X_scaled)

    AUG_X_train, AUG_X_test, AUG_y_train, AUG_y_test = train_test_split(AUG_X_pca, AUG_y_encoded, test_size=0.2, random_state=42)

    AUG_params = {'C': [0.01, 0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    AUG_model = SVC(probability=True)

    AUG_grid = GridSearchCV(AUG_model, AUG_params, cv=5, scoring='accuracy')
    AUG_grid.fit(AUG_X_train, AUG_y_train)

    AUG_best_model = AUG_grid.best_estimator_

# Eğitim sonuçları
st.success(f"✅ Model başarıyla eğitildi. Test Doğruluk: {AUG_best_model.score(AUG_X_test, AUG_y_test):.4f}")

# Karışıklık matrisi
AUG_y_pred = AUG_best_model.predict(AUG_X_test)
AUG_cm = confusion_matrix(AUG_y_test, AUG_y_pred)
fig_cm, ax_cm = plt.subplots()
disp = ConfusionMatrixDisplay(confusion_matrix=AUG_cm, display_labels=AUG_le.classes_)
disp.plot(ax=ax_cm, xticks_rotation=45)
st.pyplot(fig_cm)

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
    features = aug_extract_features_single(y, sr).reshape(1, -1)
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
