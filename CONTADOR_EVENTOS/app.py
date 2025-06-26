
import librosa
import numpy as np
import joblib
import soundfile as sf
import os
from collections import Counter
import json
from datetime import datetime  # Para timestamp

# --- 1. Configurações Globais (AJUSTE OS CAMINHOS DOS MODELOS) ---
CAMINHO_MODELO = "modelo_xgb_sono_final.pkl"  # Ajuste se necessário
CAMINHO_SCALER = "scaler_xgb_sono_final.pkl"  # Ajuste se necessário

DURACAO_SEGMENTO_SEGUNDOS = 3
SR_ALVO = 22050
PASTA_SAIDA_RECORTES = "recortes_detectados"  # Esta pasta será criada ou usada pelo script Python
OVERLAP_PERCENTAGE = 0.75
SALVAR_RECORTE_SILENCIO = True
MAX_EVENT_DURATION_SECONDS = 6
NOMES_CLASSES = ["Silêncio", "Tosse", "Respiração", "Ronco", "Ruido"]
TIMELINE_RESOLUTION_SECONDS = 0.5


# --- Funções Auxiliares (mantêm as mesmas que você tem) ---

def extract_audio_features(y, sr):
    # ... seu código extract_audio_features ...
    if len(y) < 512: return None
    try:
        n_fft = min(2048, len(y))
        if len(y) < n_fft: return None
        rms = librosa.feature.rms(y=y).mean()
        zcr = librosa.feature.zero_crossing_rate(y).mean()
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft).mean()
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft).mean()
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft).mean()
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft).mean()
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft)
        mfccs_mean = mfccs.mean(axis=1)
        mfccs_std = mfccs.std(axis=1)
        mfccs_delta = librosa.feature.delta(mfccs)
        mfccs_delta_mean = mfccs_delta.mean(axis=1)
        mfccs_delta_std = mfccs_delta.std(axis=1)
        mfccs_delta2 = librosa.feature.delta(mfccs, order=2)
        mfccs_delta2_mean = mfccs_delta2.mean(axis=1)
        mfccs_delta2_std = mfccs_delta2.std(axis=1)
        features = [rms, zcr, rolloff, centroid, bandwidth, contrast] + \
                   list(mfccs_mean) + list(mfccs_std) + \
                   list(mfccs_delta_mean) + list(mfccs_delta_std) + \
                   list(mfccs_delta2_mean) + list(mfccs_delta2_std)
        return features
    except Exception as e:
        return None


def determine_winning_classes_on_timeline(all_segment_results, total_audio_samples, sr, class_names,
                                          time_resolution_seconds):
    # ... seu código determine_winning_classes_on_timeline ...
    time_resolution_samples = int(time_resolution_seconds * sr)
    if time_resolution_samples == 0: time_resolution_samples = sr
    num_time_slices = int(np.ceil(total_audio_samples / time_resolution_samples))
    accumulated_probs_per_slice = np.zeros((num_time_slices, len(class_names)))
    for segment_info in all_segment_results:
        seg_start_slice = int(segment_info['start_sample'] / time_resolution_samples)
        seg_end_slice = int((segment_info['end_sample'] - 1) / time_resolution_samples)
        seg_start_slice = max(0, seg_start_slice)
        seg_end_slice = min(num_time_slices - 1, seg_end_slice)
        for i in range(seg_start_slice, seg_end_slice + 1):
            accumulated_probs_per_slice[i] += segment_info['probabilities']
    final_events = []
    if num_time_slices == 0: return final_events
    current_event_start_slice = 0
    silence_id = class_names.index("Silêncio") if "Silêncio" in class_names else 0
    current_event_id = silence_id
    if num_time_slices > 0 and np.sum(accumulated_probs_per_slice[0]) > 0:
        current_event_id = np.argmax(accumulated_probs_per_slice[0])
    for i in range(1, num_time_slices):
        winner_id_current_slice = silence_id
        if np.sum(accumulated_probs_per_slice[i]) > 0:
            winner_id_current_slice = np.argmax(accumulated_probs_per_slice[i])
        if winner_id_current_slice != current_event_id:
            event_start_sample = current_event_start_slice * time_resolution_samples
            event_end_sample = i * time_resolution_samples
            if event_end_sample > event_start_sample:
                final_events.append({'start_sample': event_start_sample, 'end_sample': event_end_sample,
                                     'prediction_id': current_event_id})
            current_event_start_slice = i
            current_event_id = winner_id_current_slice
    event_start_sample = current_event_start_slice * time_resolution_samples
    event_end_sample = total_audio_samples
    if event_end_sample > event_start_sample:
        final_events.append(
            {'start_sample': event_start_sample, 'end_sample': event_end_sample, 'prediction_id': current_event_id})
    return final_events


def enforce_max_event_duration(events, max_duration_samples):
    # ... seu código enforce_max_event_duration ...
    enforced_events = []
    for event in events:
        event_duration_samples = event['end_sample'] - event['start_sample']
        if event_duration_samples > max_duration_samples:
            temp_start = event['start_sample']
            while temp_start < event['end_sample']:
                temp_end = min(temp_start + max_duration_samples, event['end_sample'])
                enforced_events.append(
                    {'start_sample': temp_start, 'end_sample': temp_end, 'prediction_id': event['prediction_id']})
                temp_start = temp_end
        else:
            enforced_events.append(event)
    return enforced_events


# --- 3. Função Principal de Detecção e Segmentação ---

def process_audio_for_web(audio_input_path, output_base_folder="recortes_detectados"):
    # Carrega modelo e scaler (AQUI VAI USAR OS CAMINHOS GLOBAIS DESTE ARQUIVO)
    try:
        modelo_final = joblib.load(CAMINHO_MODELO)
        scaler_final = joblib.load(CAMINHO_SCALER)
        # print("✅ Modelo final e Scaler carregados com sucesso para processamento.") # Removido para não poluir stdout
    except FileNotFoundError as e:
        print(json.dumps({"error": f"Modelos não encontrados: {e}"}))  # Retorna erro em JSON
        return [], None
    except Exception as e:
        print(json.dumps({"error": f"Erro ao carregar modelo/scaler: {e}"}))  # Retorna erro em JSON
        return [], None

    # Carregamento e Resample do áudio
    try:
        y_long, sr_original = librosa.load(audio_input_path, sr=None)
        # print(f"\n🎧 Carregando áudio: {os.path.basename(audio_input_path)}") # Removido para não poluir stdout
    except Exception as e:
        print(json.dumps({"error": f"Erro ao carregar o áudio '{os.path.basename(audio_input_path)}': {e}"}))
        return [], None

    if sr_original != SR_ALVO:
        y_long = librosa.resample(y_long, orig_sr=sr_original, target_sr=SR_ALVO)
    sr = SR_ALVO
    total_audio_samples = len(y_long)

    # Preparação do diretório de saída para recortes de áudio
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_audio_name = os.path.basename(audio_input_path).split('.')[0]
    current_output_folder = os.path.join(output_base_folder, f"analise_{base_audio_name}_{timestamp}")
    if not os.path.exists(current_output_folder):
        os.makedirs(current_output_folder)

    # Processamento de Segmentos para Predição
    samples_per_segment = int(DURACAO_SEGMENTO_SEGUNDOS * sr)
    hop_length = int(samples_per_segment * (1 - OVERLAP_PERCENTAGE))
    if hop_length == 0: hop_length = 1

    all_segment_results_with_probs = []
    for start_sample in range(0, total_audio_samples - samples_per_segment + 1, hop_length):
        end_sample = start_sample + samples_per_segment
        segment_data = y_long[start_sample:end_sample]
        if len(segment_data) < samples_per_segment: continue
        features = extract_audio_features(segment_data, sr)
        if features is None: continue

        X_segment_scaled = scaler_final.transform([features])
        try:
            probabilities = modelo_final.predict_proba(X_segment_scaled)[0]
        except AttributeError:
            print(json.dumps({"error": "Modelo não suporta 'predict_proba'."}))
            return [], None

        all_segment_results_with_probs.append({
            'start_sample': start_sample, 'end_sample': end_sample, 'probabilities': probabilities
        })

    if not all_segment_results_with_probs:
        print(json.dumps({"message": "Nenhuma predição gerada."}))
        return [], None

    # Construção e Refinamento da Linha do Tempo de Eventos
    final_non_overlapping_events = determine_winning_classes_on_timeline(
        all_segment_results_with_probs, total_audio_samples, sr, NOMES_CLASSES, TIMELINE_RESOLUTION_SECONDS
    )
    max_duration_samples = int(MAX_EVENT_DURATION_SECONDS * sr)
    final_refined_events = enforce_max_event_duration(final_non_overlapping_events, max_duration_samples)

    # Coleta de Dados dos Eventos e Salvamento de Recortes
    events_data_list = []
    detected_classes_counts = Counter()
    event_counter_id = 0

    for event in final_refined_events:
        final_predicted_class_name = NOMES_CLASSES[event['prediction_id']]
        event_start_sample = event['start_sample']
        event_end_sample = event['end_sample']
        event_audio_data = y_long[event_start_sample:event_end_sample]

        if len(event_audio_data) == 0: continue
        detected_classes_counts[final_predicted_class_name] += 1

        # Garante que os tempos e durações são floats nativos do Python
        event_start_time = float(event_start_sample / sr)
        event_end_time = float(event_end_sample / sr)
        event_duration = float(event_end_time - event_start_time)

        event_data = {
            "id": event_counter_id + 1,
            "class": final_predicted_class_name,
            "start_time_seconds": round(event_start_time, 3),
            "end_time_seconds": round(event_end_time, 3),
            "duration_seconds": round(event_duration, 3)
        }
        events_data_list.append(event_data)

        if SALVAR_RECORTE_SILENCIO or final_predicted_class_name != "Silêncio":
            event_counter_id += 1
            output_filename = f"{final_predicted_class_name}_{int(event_start_time)}-{int(event_end_time)}s_event_{event_counter_id}.wav"
            output_filepath = os.path.join(current_output_folder, output_filename)
            try:
                sf.write(output_filepath, event_audio_data, sr)
            except Exception as e:
                # print(f"  ❌ ERRO ao salvar recorte agrupado '{output_filename}': {e}") # Evita poluir stdout
                pass  # Apenas ignora o erro de salvar recorte individual, não é crítico para o JSON

    return events_data_list, current_output_folder


# --- Execução Principal quando chamado como script ---
if __name__ == "__main__":
    # Este bloco é executado quando audio_processor.py é chamado diretamente, por exemplo, por um Node.js
    # O primeiro argumento da linha de comando será o caminho do arquivo de áudio
    import sys

    if len(sys.argv) > 1:
        audio_file_path = sys.argv[1]

        list_events, recortes_folder_path = process_audio_for_web(audio_file_path)

        # Prepara a resposta em JSON para o Node.js
        response_data = {
            "events": list_events,
            "recortes_path": recortes_folder_path
        }


    else:
        # Se for chamado sem argumentos, mostra como usar

        print("Este script deve ser chamado por um backend Node.js.")