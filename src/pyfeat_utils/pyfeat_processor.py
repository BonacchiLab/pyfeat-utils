# Importações necessárias
import os
import json
from feat import Detector, Fex
from feat.utils.io import get_test_data_path, read_feat
from feat.plotting import imshow
from IPython.core.display import Video
import matplotlib.pyplot as plt
from glob import glob
import cv2  
from IPython.display import display
import time
import pandas as pd

# Iniciar temporizador
time1 = time.perf_counter()

# Detetor do pyfeat
detector = Detector()

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
process_types = config["data_processing"].get("process_type", ["image", "video"])

# Obter todos os arquivos na pasta input_data
input_files = glob(os.path.join(data_dir, "*"))

# Processar imagens
if "image" in process_types:
    image_files = [f for f in input_files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    for image_path in image_files:
        print(f"Processing images: {image_path}")
        # Exibir a imagem
        imshow(image_path)

        # Detectar características na imagem
        prediction = detector.detect_image(image_path, data_type="image")

        # Exibir resultados
        print(prediction.head())
        print(prediction.aus)  # Action Units
        print(prediction.emotions)  # Emotions
        print(prediction.poses)  # Head pose
        print(prediction.identities)  # Identities


        # Plotar detecções com poses
        figs = prediction.plot_detections(poses=True)
        plt.show()

        
        # Guardar a saída em um arquivo CSV
        output_csv_path = os.path.join(data_dir, "output.csv")
        prediction.to_csv(output_csv_path, index=False)
        print(f"Output CSV saved to {output_csv_path} and added to input data directory.")

# Processar vídeos
if "video" in process_types:
    video_files = [f for f in input_files if f.lower().endswith((".mp4", ".avi", ".mov"))]
    for video_path in video_files:
        print(f"Processing video: {video_path}")

        # Detectar características no vídeo
        video_prediction = detector.detect_video(video_path, data_type="video", skip_frames=20)

        # Adicionar uma coluna de labels para as caras detectadas
        video_prediction['label'] = video_prediction.index.map(lambda x: f"Face_{x}")

        # Exibir as primeiras linhas do DataFrame de predição
        print(video_prediction.head())

        # Abrir o vídeo com OpenCV
        cap = cv2.VideoCapture(video_path)

        # Verificar se o vídeo foi aberto corretamente
        if not cap.isOpened():
            print(f"Could not open the video: {video_path}")
            continue
       
        # Plotar as emoções detectadas
        plt.figure(figsize=(15, 10))
        video_prediction.emotions.plot(title="Detected Emotions")
        plt.xlabel("Frames")
        plt.ylabel("Emotion Intensity")
        plt.legend(loc="upper right")
        plt.show()
        
        # Loop para processar emoções 
        for emotion in video_prediction.emotions.columns:
            filtered_frames = video_prediction[video_prediction.emotions[emotion] > 0.8]

            if not filtered_frames.empty:
                print(f"Frames with {emotion} > 0.8:")
                print(filtered_frames[['frame', emotion, 'label']])  # Inclui a label no print

                # Exibir os frames correspondentes
                for frame_number in filtered_frames['frame']:
                    # Configurar o vídeo para o frame específico
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                    ret, frame = cap.read()
                    if not ret:
                        print(f"Could not read frame {frame_number}.")
                        continue

                    # Converter o frame de BGR para RGB para exibição
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Exibir o frame
                    plt.figure(figsize=(10, 6))
                    plt.imshow(frame_rgb)
                    plt.title(f"Frame {frame_number} - {emotion} > 0.8")
                    plt.axis('off')
                    plt.show()
            else:
                print(f"No frame had {emotion} > 0.8.")

        # Libertar o vídeo
        cap.release()

        # Guardar a saída em um arquivo CSV
        output_csv_path = os.path.join(data_dir, "video_output.csv")
        video_prediction.to_csv(output_csv_path, index=False)
        #video_prediction = read_feat(data_csv_path)
        print(f"Output CSV saved to {output_csv_path} and added to input data directory.")

# Tempo de processamento de imagem e video
time2 = time.perf_counter()
print(f"Processing time: {time2 - time1} seconds")

