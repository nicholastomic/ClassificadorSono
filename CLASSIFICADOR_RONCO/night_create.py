import pandas as pd
import os
import numpy as np
import librosa
import random
from datetime import datetime, timedelta, time

# === CONFIGURAÇÕES GERAIS ===
SR = 24000
MAX_POR_PASTA = 5
CSV_DIR = "./data/csv"
AUDIO_DIR = "./data/audio"

FOLDERS = {
    "quiet": (r"quiet", 0, 100),
    "cough": (r"cough", 1, MAX_POR_PASTA),
    "breath": (r"breathe", 2, MAX_POR_PASTA),
    "snore": (r"snore", 3, MAX_POR_PASTA),
    "ruido": (r"non_wearer", 4, MAX_POR_PASTA)
}


# === EXTRAÇÃO DE FEATURES DO ÁUDIO ===
def extrair_features_y(y, sr=SR):
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
    except Exception as e:
        print(f"[!] Erro ao extrair features: {e}")
        return None


def preparar_audios(folders: dict, output_path: str):
    print("Iniciando a preparação dos áudios...")
    dados, rotulos = [], []

    for classe, (pasta, label, max_in_folder) in folders.items():
        arquivos = [f for f in os.listdir(f"{AUDIO_DIR}/{pasta}") if f.endswith(".wav")]
        random.shuffle(arquivos)
        count = 0

        for arquivo in arquivos:
            if count >= max_in_folder:
                break

            path = os.path.join(f"{AUDIO_DIR}/{pasta}", arquivo)
            y, _ = librosa.load(path, sr=SR)
            duration = len(y) / SR
            f = extrair_features_y(y, SR)

            if f:
                f.insert(0, os.path.join(pasta, arquivo))
                f.insert(1, duration)
                f.insert(2, classe)
                dados.append(f)
                rotulos.append(label)
                count += 1

    colunas = ["file", "duration", "type", "rms", "zcr", "rolloff", "centroid", "bandwidth", "contrast"] + [f"mfcc_{i}" for i in range(13)]
    df_audio = pd.DataFrame(np.array(dados), columns=colunas)
    df_audio.to_csv(output_path, index=False)

    print(f"✅ Dados de áudio salvos em {output_path}")
    return df_audio


# === PREPARAÇÃO DOS DADOS DE AMBIENTE ===

def get_night_date(dt):
    if dt.time() < time(12, 0):
        return (dt - timedelta(days=1)).date()
    return dt.date()


def is_sleep_time_generator(hora_dormir, hora_acordar):
    def is_sleep_time(dt):
        t = dt.time()
        return t >= hora_dormir or t <= hora_acordar
    return is_sleep_time


def apply_sleep_filter(group):
    start_hour = random.randint(18, 22)
    end_hour = random.randint(5, 10)
    start_minute = random.randint(0, 59)
    end_minute = random.randint(0, 59)

    hora_dormir = time(start_hour, start_minute)
    hora_acordar = time(end_hour, end_minute)

    return group[group['date_time'].apply(is_sleep_time_generator(hora_dormir, hora_acordar))]


def preparar_dados_ambiente(path_csv: str, output_path: str):
    print("Iniciando a preparação dos dados do ambiente...")
    df = pd.read_csv(path_csv)
    df['date_time'] = pd.to_datetime(df['date_time'])
    df['night_date'] = df['date_time'].apply(get_night_date)

    df_noite = (
        df
        .set_index('night_date')
        .groupby(level=0, group_keys=False)
        .apply(apply_sleep_filter)
        .reset_index()
    )

    df_noite = df_noite.sort_values('date_time').reset_index(drop=True)
    df_noite['delta'] = df_noite['date_time'].diff()
    df_noite['block_id'] = (df_noite['delta'] > timedelta(minutes=2)).cumsum()
    df_noite = df_noite.drop(columns='delta')

    min_duration = timedelta(hours=4)
    valid_blocks = (
        df_noite.groupby('block_id')
        .filter(lambda g: g['date_time'].max() - g['date_time'].min() >= min_duration)
    )

    valid_blocks_clean = valid_blocks[["night_date", "date_time", "humidity", "light", "temperature"]]
    valid_blocks_clean.to_csv(output_path, index=True)

    print(valid_blocks_clean.head())
    print(f"✅ Dados do ambiente salvos em {output_path}")
    return valid_blocks_clean

# === GERAÇÃO DO DF FINAL ===
def gerar_df_final(df_audio, df_ambiente, output_path):
    print("Gerando DataFrame final...")

    # Bagunça os dados de áudio
    shuffled_audio = df_audio.sample(frac=1, random_state=42).reset_index(drop=True)

    shuffled_index = 0
    total_audio = len(shuffled_audio)

    df_ambiente = df_ambiente.head(100)

    # Percorre o df de ambiente
    for i in range(len(df_ambiente) - 1):
        actual_env = df_ambiente.iloc[i]
        next_env = df_ambiente.iloc[i + 1]
        
        env_diff = next_env.date_time - actual_env.date_time

        if env_diff > timedelta(seconds=90):  # tolerância de 90 segundos (você comentou 30 na linha, mas usou 90)
            env_duration = timedelta(minutes=1)
        else:
            env_duration = env_diff

        duration = timedelta(0)

        while duration <= env_duration and shuffled_index < total_audio:
            # Modifica direto no DataFrame usando .at
            shuffled_audio.at[shuffled_index, 'night_date'] = actual_env.night_date
            shuffled_audio.at[shuffled_index, 'date_time'] = actual_env.date_time + duration

            # Incrementa índice para próxima linha de áudio
            shuffled_index += 1

            # Adiciona a duração do áudio atual (garanta que 'duration' esteja em segundos)
            duration_seconds = float(shuffled_audio.at[shuffled_index - 1, 'duration'])
            duration += timedelta(seconds=duration_seconds)

    # Remove linhas com date_time/night_data vazio ou NaN
    shuffled_audio = shuffled_audio.dropna(subset=['date_time']).reset_index(drop=True)
    shuffled_audio = shuffled_audio.dropna(subset=['night_date']).reset_index(drop=True)

    # Salvar ou retornar
    shuffled_audio.to_csv(output_path, index=False)
    print(f"Arquivo salvo em {output_path}")

    return shuffled_audio

# === EXECUÇÃO PRINCIPAL ===

if __name__ == "__main__":
    os.makedirs(CSV_DIR, exist_ok=True)

    df_audio = preparar_audios(FOLDERS, f"{CSV_DIR}/dados_treino_teste.csv")
    df_ambiente = preparar_dados_ambiente("rpi_20_plus.csv", f"{CSV_DIR}/dados_ambiente.csv")

    # df_audio = pd.read_csv(f"{CSV_DIR}/dados_treino_teste.csv")
    # df_audio = df_audio.dropna().reset_index(drop=True)
    # df_ambiente = pd.read_csv(f"{CSV_DIR}/dados_ambiente.csv")
    # df_ambiente = df_ambiente.dropna().reset_index(drop=True)
    # df_ambiente["date_time"] = pd.to_datetime(df_ambiente["date_time"])
    df_final = gerar_df_final(df_audio, df_ambiente, f"{CSV_DIR}/dados_final.csv")
    # df_ambiente = df_ambiente.head(100)
    # df_ambiente.to_csv("dados_ambiente.csv", index=False)
    print("✅ Preparação concluída com sucesso!")
