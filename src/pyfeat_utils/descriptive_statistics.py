import os
import pandas as pd
import json
import time
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns

# Iniciar temporizador
time1 = time.perf_counter()

# Caminho para a config
config_path = os.path.join(os.path.dirname(__file__), "template_config.json")
if not os.path.exists(config_path):
    raise FileNotFoundError(f"'template_config.json' was not found in {config_path}")

# Carregar configurações do arquivo template_config.json
with open(config_path, "r") as config_file:
    config = json.load(config_file)

# Caminho correto para a pasta de dados, expandindo ~ para o diretório do usuário
data_dir = os.path.expanduser(config["data_processing"]["data_pyfeat-utils"])

# Verificar se a pasta existe
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"'data_pyfeat-utils' was not found at {data_dir}")

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

    # Most common emotion
    most_common, emotion_counts = get_most_common_emotion(df, emotion_columns)
    print(f"Most common emotion: {most_common}")
    print("Emotion distribution:")
    print(emotion_counts)

    # Mean level of the most common emotion(s)
    print("\n=== Mean Level of Most Common Emotion(s) ===")
    mask = df[emotion_columns].idxmax(axis=1) == most_common
    mean_level = df.loc[mask, most_common].mean()
    print(f"Mean level for '{most_common}' (when it is the highest): {mean_level:.3f}")

    # Median, mean, and quartiles for each emotion
    print("\n=== Mean, Median, and Quartiles for Each Emotion ===")
    for emotion in emotion_columns:
        print(f"{emotion}: mean={df[emotion].mean():.3f}, median={df[emotion].median():.3f}, Q1={df[emotion].quantile(0.25):.3f}, Q3={df[emotion].quantile(0.75):.3f}")


## PLOTS ##

    # Plot emotion distribution
    plt.figure(figsize=(8, 5))
    sns.barplot(x=emotion_counts.index, y=emotion_counts.values, palette="viridis")
    plt.title("Emotion Distribution (Most Common per Row)")
    plt.ylabel("Count")
    plt.xlabel("Emotion")
    plt.tight_layout()
    plt.show()


    # Boxplot for each emotion
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df[emotion_columns], palette="Set2")
    plt.title("Boxplot of Emotion Levels")
    plt.ylabel("Level")
    plt.xlabel("Emotion")
    plt.tight_layout()
    plt.show()

    # Barplot for mean and median of each emotion
    means = df[emotion_columns].mean()
    medians = df[emotion_columns].median()
    stats_df = pd.DataFrame({'mean': means, 'median': medians})

    stats_df.plot(kind='bar', figsize=(10, 5))
    plt.title("Mean and Median for Each Emotion")
    plt.ylabel("Value")
    plt.xlabel("Emotion")
    plt.tight_layout()
    plt.show()

    # Detect AU columns (e.g., AU01, AU02, ..., AU66)
    au_columns = [col for col in df.columns if col.startswith("AU") and pd.api.types.is_numeric_dtype(df[col])]
    if au_columns:
        print("\n=== Mean, Median, and Quartiles for Each AU ===")
        for au in au_columns:
            print(f"{au}: mean={df[au].mean():.3f}, median={df[au].median():.3f}, Q1={df[au].quantile(0.25):.3f}, Q3={df[au].quantile(0.75):.3f}")

        # Boxplot for each AU
        plt.figure(figsize=(max(10, len(au_columns) // 2), 6))
        sns.boxplot(data=df[au_columns], palette="coolwarm")
        plt.title("Boxplot of AU Levels")
        plt.ylabel("Level")
        plt.xlabel("AU")
        plt.tight_layout()
        plt.show()

        # Barplot for mean and median of each AU
        au_means = df[au_columns].mean()
        au_medians = df[au_columns].median()
        au_stats_df = pd.DataFrame({'mean': au_means, 'median': au_medians})

        au_stats_df.plot(kind='bar', figsize=(max(10, len(au_columns) // 2), 5))
        plt.title("Mean and Median for Each AU")
        plt.ylabel("Value")
        plt.xlabel("AU")
        plt.tight_layout()
        plt.show()
    else:
        print("\nNo AU columns found in the files.")


if __name__ == "__main__":
    main()


# Processing time
time2 = time.perf_counter()
print(f"Processing time: {time2 - time1} seconds")
