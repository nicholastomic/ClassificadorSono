import os
import numpy as np
import librosa
import random
import joblib
import scipy.signal
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
# === CONFIGURAÇÕES ===
sr = 22050
max_por_pasta = 1000

# === PASTAS COM OS ÁUDIOS ===
pastas = {
    "quiet": (r"C:\Users\Nicholas\OneDrive\Desktop\DATASET DE RUIDO SONO\train\home\v-wangzeyu\skywang\DreamCatcher_cropped\data\train\quiet", 0),
    "cough": (r"C:\Users\Nicholas\OneDrive\Desktop\DATASET DE RUIDO SONO\train\home\v-wangzeyu\skywang\DreamCatcher_cropped\data\train\cough", 1),
    "breath": (r"C:\Users\Nicholas\OneDrive\Desktop\DATASET DE RUIDO SONO\train\home\v-wangzeyu\skywang\DreamCatcher_cropped\data\train\breathe", 2),
    "snore": (r"C:\Users\Nicholas\OneDrive\Desktop\DATASET DE RUIDO SONO\train\home\v-wangzeyu\skywang\DreamCatcher_cropped\data\train\snore", 3)
}
# === EXTRAÇÃO DE FEATURES AMPLIADAS ===
def extrair_features_y(y, sr=22050):
    if len(y) < 512:
        return None
    try:
        n_fft = min(2048, len(y))
        rms = librosa.feature.rms(y=y).mean()
        zcr = librosa.feature.zero_crossing_rate(y).mean()
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft).mean(axis=1)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr).mean()
        return [rms, zcr, rolloff, centroid, bandwidth, contrast] + list(mfccs)
    except:
        return None

# === LEITURA E EXTRAÇÃO ===
dados, rotulos = [], []

for classe, (pasta, label) in pastas.items():
    arquivos = [f for f in os.listdir(pasta) if f.endswith(".wav")]
    random.shuffle(arquivos)
    count = 0

    for arquivo in arquivos:
        if count >= max_por_pasta:
            break

        path = os.path.join(pasta, arquivo)
        y, _ = librosa.load(path, sr=sr)
        f = extrair_features_y(y, sr)
        if f:
            dados.append(f)
            rotulos.append(label)
            count += 1

print(f"Total de amostras: {len(dados)}")

# === PRÉ-PROCESSAMENTO + SMOTE ===
# Utilização de smote (tecnica de criação de dados sinteticos com base em dados reais) na classe "Tosse"
# para compensar o desbalanceamento de classes
X = np.array(dados)
y = np.array(rotulos)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_scaled, y)
print(f"Amostras após SMOTE: {len(y_res)}")

# === TREINAMENTO COM XGBoost (Modelo de IA) ===
modelo = XGBClassifier(n_estimators=100, random_state=42)
scores = cross_val_score(modelo, X_res, y_res, cv=5)

print("\nAcurácias por fold:", scores)
print("Acurácia média:", scores.mean())

y_pred = cross_val_predict(modelo, X_res, y_res, cv=5)
print(confusion_matrix(y_res, y_pred))
print(classification_report(y_res, y_pred, target_names=["Silêncio", "Tosse", "Respiração", "Ronco"]))

# === SALVANDO MODELO E SCALER ===
# modelo.fit(X_res, y_res)
# joblib.dump(modelo, "modelo_xgb_tosse.pkl")
# joblib.dump(scaler, "scaler_xgb.pkl")
# print("\n✅ Modelo e scaler salvos com sucesso.")