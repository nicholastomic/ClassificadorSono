import pandas as pd
import os
import numpy as np
import librosa
import random
from datetime import datetime, timedelta, time

sr = 24000
max_por_pasta = 5

CSV_DIR = "./data/csv"

folders = {
    "quiet": (r"./data/audio/quiet", 0, 100),
    "cough": (r"./data/audio/cough", 1, max_por_pasta),
    "breath": (r"./data/audio/breathe", 2, max_por_pasta),
    "snore": (r"./data/audio/snore", 3, max_por_pasta)
}

# print("Iniciando a preparação dos áudios...")

# def extrair_features_y(y, sr=24000):
#     if len(y) < 512:
#         return None
#     try:
#         n_fft = min(2048, len(y))
#         rms = librosa.feature.rms(y=y).mean()
#         zcr = librosa.feature.zero_crossing_rate(y).mean()
#         mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft).mean(axis=1)
#         rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
#         centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
#         bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
#         contrast = librosa.feature.spectral_contrast(y=y, sr=sr).mean()
#         return [rms, zcr, rolloff, centroid, bandwidth, contrast] + list(mfccs)
#     except:
#         return None
    
# dados, rotulos = [], []

# for classe, (pasta, label, max_in_folder) in folders.items():
#     arquivos = [f for f in os.listdir(pasta) if f.endswith(".wav")]
#     random.shuffle(arquivos)
#     count = 0

#     for arquivo in arquivos:
#         if count >= max_in_folder:
#             break

#         path = os.path.join(pasta, arquivo)
#         y, _ = librosa.load(path, sr=sr)
#         duration = len(y) / sr
#         f = extrair_features_y(y, sr)
#         f.insert(0, path)
#         f.insert(1, duration)
#         if f:
#             dados.append(f)
#             rotulos.append(label)
#             count += 1

# print(f"Total de amostras de aúdio: {len(dados)}")

# X = np.array(dados)

# colunas = ["file", "duration", "rms", "zcr", "rolloff", "centroid", "bandwidth", "contrast"] + [f"mfcc_{i}" for i in range(13)]
# df_ambiente = pd.DataFrame(X, columns=colunas)

# df_ambiente.to_csv(f"{CSV_DIR}/dados_treino_teste.csv", index=False)
# print(f"✅ Dados salvos em {CSV_DIR}/dados_treino_teste.csv")

## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ##

print("Iniciando a preparação dos dados do ambiente...")
df_ambiente = pd.read_csv("rpi_20_plus.csv")
df_ambiente['date_time'] = pd.to_datetime(df_ambiente['date_time'])

# Criar uma nova coluna com a "data base da noite"
def get_night_date(dt):
    if dt.time() < time(12, 0):  # se for de madrugada (antes do meio-dia), considera noite do dia anterior
        return (dt - timedelta(days=1)).date()
    else:
        return dt.date()

df_ambiente['night_date'] = df_ambiente['date_time'].apply(get_night_date)

# Função para aplicar intervalo noturno para cada grupo
def apply_sleep_filter(group):
    # Define horário aleatório de dormir e acordar para aquela noite
    start_hour = random.randint(18, 22)
    end_hour = random.randint(5, 10)
    start_minute = random.randint(0, 59)
    end_minute = random.randint(0, 59)

    hora_dormir = time(start_hour, start_minute)
    hora_acordar = time(end_hour, end_minute)

    # Filtra os dados do grupo conforme o horário definido
    def is_sleep_time(dt):
        t = dt.time()
        return t >= hora_dormir or t <= hora_acordar  # noite cruzando meia-noite

    return group[group['date_time'].apply(is_sleep_time)]

# Aplica o filtro para cada noite
# df_ambiente_noite = df_ambiente.groupby('night_date', group_keys=False, include_groups=True).apply(apply_sleep_filter)
df_ambiente_noite = (
    df_ambiente
    .set_index('night_date')
    .groupby(level=0, group_keys=False)
    .apply(apply_sleep_filter)
    .reset_index()
)

# Continua o mesmo processo anterior...
df_ambiente_noite = df_ambiente_noite.sort_values('date_time').reset_index(drop=True)
df_ambiente_noite['delta'] = df_ambiente_noite['date_time'].diff()
df_ambiente_noite['block_id'] = (df_ambiente_noite['delta'] > timedelta(minutes=2)).cumsum()
df_ambiente_noite = df_ambiente_noite.drop(columns='delta')

# Mantém apenas blocos com pelo menos 4 horas de dados
min_duration = timedelta(hours=4)
valid_blocks = (
    df_ambiente_noite.groupby('block_id')
    .filter(lambda g: g['date_time'].max() - g['date_time'].min() >= min_duration)
)

valid_blocks_clean = valid_blocks[["night_date", "date_time", "humidity", "light", "temperature"]]
valid_blocks_clean.to_csv(f"{CSV_DIR}/dados_ambiente.csv", index=False)

print(f"✅ Dados do ambiente salvos em {CSV_DIR}/dados_ambiente.csv")
print("Preparação concluída com sucesso!")