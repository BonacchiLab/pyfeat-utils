# Importações necessárias
import os
import json
from feat import Detector, Fex
from feat.utils.io import get_test_data_path, read_feat
from feat.plotting import imshow
from IPython.core.display import Video
import matplotlib.pyplot as plt
from glob import glob
import cv2  # Biblioteca para manipulação de vídeo
from IPython.display import display
import time

time1 = time.perf_counter()

# Inicializar o detector
detector = Detector()

# Caminho relativo para a pasta de entrada
input_data_dir = os.path.join(os.path.dirname(__file__), "input_data")

output_data_dir = os.path.join(os.path.dirname(__file__), "output_data")

# Verificar se a pasta existe
if not os.path.exists(input_data_dir):
    raise FileNotFoundError(f"'input_data' was not {input_data_dir}")

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
input_files = glob(os.path.join(input_data_dir, "*"))

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
        print(prediction.emotions.plots())

        # Plotar detecções com poses
        figs = prediction.plot_detections(poses=True)
        plt.show()

        # Plotar as emoções detectadas
        plt.figure(figsize=(15, 10))
        prediction.emotions.plot(title="Detected Emotions")
        plt.xlabel("Frames")
        plt.ylabel("Emotion Intensity")
        plt.legend(loc="upper right")
        plt.show()
        
        # Guardar a saída em um arquivo CSV
        """output_dir = config["data_processing"]["output_data"]
        os.makedirs(output_dir, exist_ok=True)  # Garante que diretório exista
        # Salvar o DataFrame em um arquivo CSV
        output_csv_path = os.path.join(output_dir, "output.csv")
        prediction.to_csv(output_csv_path, index=False)"""

# Processar vídeos
if "video" in process_types:
    video_files = [f for f in input_files if f.lower().endswith((".mp4", ".avi", ".mov"))]
    for video_path in video_files:
        print(f"Processing videos: {video_path}")
        # Abrir o vídeo com OpenCV
        cap = cv2.VideoCapture(video_path)  

        # Verificar se o vídeo foi aberto corretamente
        if not cap.isOpened():
            print(f"Could not open the video: {video_path}")
            continue

        # Redefinir o índice do DataFrame para evitar ambiguidade
        video_prediction = detector.detect_video(video_path, data_type="video", skip_frames=20)

        # Exibir o vídeo
        display(Video(video_path, embed=True))

        # Exibir resultados
        print(video_prediction.head())
        print(prediction.emotions.plots())

        # Plotar emoções ao longo do vídeo
        plt.figure(figsize=(15, 10))
        axes = video_prediction.emotions.plot(title="Emotions throughout the video")
        plt.show()

        # Exibir as emoções detectadas ao longo do vídeo
        plt.figure(figsize=(15, 10))
        video_prediction.emotions.plot(title="Emotions throughout the video")
        plt.xlabel("Frames")
        plt.ylabel("Emotion Intensity")
        plt.legend(loc="upper right")
        plt.show()

        # Visualizar detecções no vídeo
        figs = video_prediction.plot_detections(poses=True)
        plt.show()

        # Loop imprimir frames com AUs > 0.8 para cada emoção
        for emotion in video_prediction.emotions.columns:
            filtered_frames = video_prediction[video_prediction.emotions[emotion] > 0.8]

            if not filtered_frames.empty:
                print(f"Frames with {emotion} > 0.8:")
                print(filtered_frames[['frame', emotion]])  # Display the frames and the emotion value

                # Exibir os frames filtrados
                max_frames_to_process = 5  # Limitar o número de frames processados
                for i, frame_number in enumerate(filtered_frames['frame']):
                    if i >= max_frames_to_process:
                        print("Frame processing limit reached.")
                        break
                    # Verificar se o frame está dentro do alcance do vídeo
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if frame_number >= total_frames:
                        print(f"{frame_number} is out of video range.")
                        continue
                    # Configurar o vídeo para o frame específico
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                    ret, frame = cap.read()
                    if not ret:
                        print(f"The frame {frame_number} was not captured.")
                        continue
                    # Converter o frame de BGR para RGB (para exibição com matplotlib)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Exibir o frame como imagem.
                    plt.figure(figsize=(20, 16))
                    plt.imshow(frame_rgb)
                    plt.title(f"Frame {frame_number} - {emotion} > 0.8")
                    plt.axis('off')
                    plt.pause(0.001)  # Exibir sem bloquear o loop
                    plt.close('all')  # Fechar a figura para liberar memória
            else:
                print(f"No frame had {emotion} > 0.8.")
        # Libertar o vídeo
        cap.release()
        
        """output_csv_path = os.path.join(output_data_dir, "output.csv")
        prediction.to_csv(output_csv_path, index=False)"""

# Tempo de processamento de imagem e video
time2 = time.perf_counter()
print(f"Processing time for image and videos: {time2 - time1} seconds")

