import os
import pandas as pd
import json
import time
from glob import glob

# Iniciar temporizador
time1 = time.perf_counter()

# Caminho 
data_dir = os.path.join(os.path.dirname(__file__), "input_data")

# Verificar se a pasta existe
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"'input_data' was not {data_dir}")

# Caminho relativo para o arquivo template_config.json
config_path = os.path.join(os.path.dirname(__file__), "template_config.json")
if not os.path.exists(config_path):
    raise FileNotFoundError(f"'template_config.json' was not found in {config_path}")

# Carregar configurações do arquivo template_config.json
with open(config_path, "r") as config_file:
    config = json.load(config_file)

# Determinar se o script deve processar imagens, vídeos ou ambos
process_types = config["data_processing"].get("process_type", ["csv", "txt"])

# Obter todos os arquivos na pasta input_data
input_files = glob(os.path.join(data_dir, "*"))

def load_files_from_dir(directory, extensions=(".csv", ".txt")):
    files = [os.path.join(directory, f) for f in os.listdir(directory)
             if f.lower().endswith(extensions)]
    dfs = []
    for file in files:
        try:
            if file.endswith(".csv"):
                df = pd.read_csv(file)
            elif file.endswith(".txt"):
                # Tenta ler como tab separado, se falhar tenta vírgula
                try:
                    df = pd.read_csv(file, delimiter="\t")
                except Exception:
                    df = pd.read_csv(file, delimiter=",")
            else:
                continue
            dfs.append(df)
        except Exception as e:
            print(f"Could not read {file}: {e}")
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        return pd.DataFrame()

def get_most_common_emotion(df, emotion_columns):
    most_common = df[emotion_columns].idxmax(axis=1)
    return most_common.value_counts().idxmax(), most_common.value_counts()

def main():
    # Carregar todos os ficheiros CSV e TXT do diretório definido no JSON
    df = load_files_from_dir(data_dir, extensions=(".csv", ".txt"))
    if df.empty:
        print("No CSV or TXT files found or files are empty.")
        return

    # Detectar colunas de emoção numéricas
    possible_emotions = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]
    emotion_columns = [col for col in df.columns if col in possible_emotions and pd.api.types.is_numeric_dtype(df[col])]
    if not emotion_columns:
        print("No numeric emotion columns found in the files.")
        return

    print("=== General Statistics ===")
    print(f"Total rows: {len(df)}")
    print(f"Emotion columns: {emotion_columns}")

    # Emoção mais comum
    most_common, emotion_counts = get_most_common_emotion(df, emotion_columns)
    print(f"Most common emotion: {most_common}")
    print("Emotion distribution:")
    print(emotion_counts)

    # Mediana, média e quartis para cada emoção
    print("\n=== Median, Mean, and Quartiles for Each Emotion ===")
    for emotion in emotion_columns:
        print(f"{emotion}: median={df[emotion].median():.3f}, mean={df[emotion].mean():.3f}, Q1={df[emotion].quantile(0.25):.3f}, Q3={df[emotion].quantile(0.75):.3f}")

    # Tempo médio (se existir approx_time)
    if "approx_time" in df.columns and pd.api.types.is_numeric_dtype(df["approx_time"]):
        avg_time = df["approx_time"].mean()
        print(f"\nAverage approx_time per row: {avg_time:.3f} seconds")
        print(f"Total duration (approx): {df['approx_time'].max():.3f} seconds")
    else:
        print("\nNo approx_time column found.")

    # Guardar estatísticas num CSV
    stats = {
        "emotion": emotion_columns,
        "median": [df[e].median() for e in emotion_columns],
        "mean": [df[e].mean() for e in emotion_columns],
        "Q1": [df[e].quantile(0.25) for e in emotion_columns],
        "Q3": [df[e].quantile(0.75) for e in emotion_columns],
        "most_common": [most_common if e == most_common else "" for e in emotion_columns]
    }
    stats_df = pd.DataFrame(stats)
    stats_csv_path = os.path.join(data_dir, "statistics_summary.csv")
    stats_df.to_csv(stats_csv_path, index=False)
    print(f"\nStatistics summary saved to {stats_csv_path}")

if __name__ == "__main__":
    main()

# Tempo de processamento
time2 = time.perf_counter()
print(f"Processing time: {time2 - time1} seconds")
