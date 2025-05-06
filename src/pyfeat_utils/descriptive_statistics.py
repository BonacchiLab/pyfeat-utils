import os

# Estatísticas descritivas para vídeos
for video_path in video_files:
    video_name = os.path.basename(video_path)
    print(f"\nDescriptive statistics for video: {video_name}")

    # Extrair emoções do objeto py-feat
    emotions_data = video_prediction.emotions

    # Calcular estatísticas descritivas
    descriptive_stats = emotions_data.describe()
    print("\nDescriptive statistics for emotions:")
    print(descriptive_stats)

    # Calcular mediana e quartis
    mediana = emotions_data.median()
    quartis = emotions_data.quantile([0.25, 0.5, 0.75])
    print("\nMedian:")
    print(mediana)
    print("\nQuartiles:")
    print(quartis)

    # Perguntar ao usuário se deseja salvar as estatísticas
    ans = input("Do you want to save the descriptive statistics? (y/n): ").strip().lower()
    if ans == "y":
        # Criar diretório de saída, se necessário
        os.makedirs(output_data_dir, exist_ok=True)

        # Salvar estatísticas descritivas em um arquivo CSV
        output_csv_path = os.path.join(output_data_dir, f"{video_name}_descriptive_statistics.csv")
        descriptive_stats.to_csv(output_csv_path)
        print(f"Descriptive statistics saved to {output_csv_path}.")
    else:
        print("Descriptive statistics not saved.")
