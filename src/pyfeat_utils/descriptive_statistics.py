import os
import pandas as pd
import json
import time
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns

# Start timer
start_time = time.perf_counter()

# Path to config
config_path = os.path.join(os.path.dirname(__file__), "template_config.json")
if not os.path.exists(config_path):
    raise FileNotFoundError(f"'template_config.json' was not found in {config_path}")

# Load configuration from template_config.json
with open(config_path, "r") as config_file:
    config = json.load(config_file)

# Correct path to data folder, expanding ~ to user directory
data_dir = os.path.expanduser(config["data_processing"]["data_pyfeat-utils"])

# Check if folder exists
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"'data_pyfeat-utils' was not found at {data_dir}")

# Determine if the script should process images, videos, or both
process_types = config["data_processing"].get("process_type", ["csv", "txt"])

# Get all files in input_data folder
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
                # Try tab delimiter, if it fails try comma
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

def load_au_translator(translator_path):
    au_map = {}
    with open(translator_path, encoding="utf-8") as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) >= 3:
                au_code = parts[0]
                muscle = parts[2]
                au_map[au_code] = muscle
    return au_map

def main():
    # Load all CSV and TXT files from the directory defined in the JSON
    files_to_analyze = [os.path.join(data_dir, f) for f in os.listdir(data_dir)
                        if f.lower().endswith((".csv", ".txt"))]
    
    # Ignore CSVs that contain video extensions in the name
    video_exts = (".jpg", ".png", ".jpeg")
    files_to_analyze = [
        f for f in files_to_analyze
        if not any(ext in os.path.basename(f).lower() for ext in video_exts)
    ]

    if not files_to_analyze:
        print("No CSV or TXT files found or files are empty.")
        return

    print("Files being analyzed:")
    for f in files_to_analyze:
        print(f"  - {os.path.basename(f)}")

    # Load AU translator
    translator_path = os.path.join(os.path.dirname(__file__), "AUs_translator.txt")
    au_translator = load_au_translator(translator_path)

    for file_path in files_to_analyze:
        print(f"\n=== Statistics for file: {os.path.basename(file_path)} ===")
        df = load_files_from_dir(os.path.dirname(file_path), extensions=(os.path.splitext(file_path)[1],))
        # Filter only the current file
        df = df[df.columns.intersection(pd.read_csv(file_path, nrows=0).columns)]
        if df.empty:
            print("File is empty or could not be read.")
            continue

        possible_emotions = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]
        emotion_columns = [col for col in df.columns if col in possible_emotions and pd.api.types.is_numeric_dtype(df[col])]
        if not emotion_columns:
            print("No numeric emotion columns found in the file.")
            continue

        print(f"Total rows: {len(df)}")
        print(f"Emotion columns: {emotion_columns}")

        most_common, emotion_counts = get_most_common_emotion(df, emotion_columns)
        print(f"Most common emotion: {most_common}")
        print("Emotion distribution:")
        print(emotion_counts)

        print("\n=== Mean Level of Most Common Emotion(s) ===")
        mask = df[emotion_columns].idxmax(axis=1) == most_common
        mean_level = df.loc[mask, most_common].mean()
        print(f"Mean level for '{most_common}' (when it is the highest): {mean_level:.3f}")

        print("\n=== Mean, Median, and Quartiles for Each Emotion ===")
        for emotion in emotion_columns:
            print(f"{emotion}: mean={df[emotion].mean():.3f}, median={df[emotion].median():.3f}, Q1={df[emotion].quantile(0.25):.3f}, Q3={df[emotion].quantile(0.75):.3f}")

        print("\n=== Times/Frames Where Each Emotion Was Most Present ===")
        for emotion in emotion_columns:
            max_val = df[emotion].max()
            max_rows = df[df[emotion] == max_val]
            for _, row in max_rows.iterrows():
                frame = row["frame"]
                print(f"Emotion '{emotion}' most present at frame {frame} (value={max_val:.3f})")

        # AU columns
        au_columns = [col for col in df.columns if col.startswith("AU") and pd.api.types.is_numeric_dtype(df[col])]
        if au_columns:
            print("\n=== Mean, Median, and Quartiles for Each AU ===")
            for au in au_columns:
                print(f"{au}: mean={df[au].mean():.3f}, median={df[au].median():.3f}, Q1={df[au].quantile(0.25):.3f}, Q3={df[au].quantile(0.75):.3f}")
        else:
            print("\nNo AU columns found in the file.")

        #PLOTS 
        if emotion_columns:
            plt.figure(figsize=(8, 5))
            sns.barplot(x=emotion_counts.index, y=emotion_counts.values, palette="viridis")
            plt.title(f"Emotion Distribution (Most Common per Row) - {os.path.basename(file_path)}")
            plt.ylabel("Count")
            plt.xlabel("Emotion")
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(10, 6))
            sns.boxplot(data=df[emotion_columns], palette="Set2")
            plt.title(f"Boxplot of Emotion Levels - {os.path.basename(file_path)}")
            plt.ylabel("Level")
            plt.xlabel("Emotion")
            plt.tight_layout()
            plt.show()

            means = df[emotion_columns].mean()
            medians = df[emotion_columns].median()
            stats_df = pd.DataFrame({'mean': means, 'median': medians})
            stats_df.plot(kind='bar', figsize=(10, 5))
            plt.title(f"Mean and Median for Each Emotion - {os.path.basename(file_path)}")
            plt.ylabel("Value")
            plt.xlabel("Emotion")
            plt.tight_layout()
            plt.show()

        if au_columns:
            # AU labels 
            au_labels = [au_translator.get(au, au) for au in au_columns]

            plt.figure(figsize=(max(10, len(au_columns) // 2), 6))
            sns.boxplot(data=df[au_columns])
            plt.title(f"Boxplot of AU Levels - {os.path.basename(file_path)}")
            plt.ylabel("Level")
            plt.xlabel("AU")
            plt.xticks(ticks=range(len(au_labels)), labels=au_labels, rotation=45, ha="right", fontsize=8)
            plt.tight_layout()
            plt.show()

            au_means = df[au_columns].mean()
            au_medians = df[au_columns].median()
            au_stats_df = pd.DataFrame({'mean': au_means, 'median': au_medians})
            au_stats_df.index = [au_translator.get(au, au) for au in au_stats_df.index]
            au_stats_df.plot(kind='bar', figsize=(max(10, len(au_columns) // 2), 5))
            plt.title(f"Mean and Median for Each AU - {os.path.basename(file_path)}")
            plt.ylabel("Value")
            plt.xlabel("AU")
            plt.xticks(ticks=range(len(au_labels)), labels=au_labels, rotation=45, ha="right", fontsize=8)
            plt.tight_layout()
            plt.show()
      

   


if __name__ == "__main__":
    main()


# Processing time
end_time = time.perf_counter()
print(f"Processing time: {end_time - start_time} seconds")