import pandas as pd
import os
import numpy as np
import librosa
import random
from datetime import datetime, timedelta, time

sr = 24000
max_por_pasta = 5

folders = {
    "quiet": (r"./data/quiet", 0, 100),
    "cough": (r"./data/cough", 1, max_por_pasta),
    "breath": (r"./data/breathe", 2, max_por_pasta),
    "snore": (r"./data/snore", 3, max_por_pasta)
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

# df_ambiente.to_csv("dados_treino_teste.csv", index=False)
# print("✅ Dados salvos em dados_treino_teste.csv")

## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ##
print("Iniciando a preparação dos dados do ambiente...")

df_ambiente = pd.read_csv("rpi_20_plus.csv")
df_ambiente['date_time'] = pd.to_datetime(df_ambiente['date_time'])

# Ordena por data
df_ambiente = df_ambiente.sort_values('date_time').reset_index(drop=True)

# Calcula diferença para definir blocos (gap > 2 min)
df_ambiente['delta'] = df_ambiente['date_time'].diff()
df_ambiente['block_id'] = (df_ambiente['delta'] > timedelta(minutes=2)).cumsum()

# Remove coluna auxiliar
df_ambiente = df_ambiente.drop(columns='delta')

min_duration = timedelta(hours=4)

blocks = []

for block_id, group in df_ambiente.groupby('block_id'):
    duration = group['date_time'].max() - group['date_time'].min()
    if duration < min_duration:
        # Ignora blocos muito curtos
        continue

    # Gera um horário inicial aleatório noturno entre 18h e 22h
    start_hour = random.randint(18, 22)
    start_minute = random.randint(0, 59)
    start_time = time(start_hour, start_minute)

    # Gera um horário final aleatório entre 4h e 10h
    # Pode ser depois da meia-noite, então lógica circular
    end_hour = random.randint(4, 10)
    end_minute = random.randint(0, 59)
    end_time = time(end_hour, end_minute)

    # Função que verifica se um horário está dentro do intervalo noturno desse bloco
    def is_in_block_night(t, start=start_time, end=end_time):
        if start <= end:
            return start <= t <= end
        else:
            # intervalo passa da meia-noite
            return t >= start or t <= end

    # Aplica o filtro para manter só as linhas do bloco que estão dentro do intervalo gerado
    filtered_group = group[group['date_time'].dt.time.apply(is_in_block_night)]

    # Adiciona coluna para indicar qual é a "noite"
    filtered_group = filtered_group.copy()
    filtered_group['night_block'] = block_id  # pode usar um contador também se quiser

    # Guarda o grupo filtrado se não vazio
    if not filtered_group.empty:
        blocks.append(filtered_group)

# Concatena todos os blocos filtrados
result = pd.concat(blocks).sort_values('date_time')

# Salva o resultado final
result[["date_time", "humidity", "light", "temperature", "night_block"]].to_csv("dados_ambiente.csv", index=False)

print("✅ Dados do ambiente salvos em dados_ambiente.csv")
print("Preparação concluída com sucesso!")