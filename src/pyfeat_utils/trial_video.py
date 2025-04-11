# Importações necessárias
import os
import matplotlib.pyplot as plt
from feat import Detector, Fex
from feat.utils.io import get_test_data_path
from feat.plotting import imshow
from IPython.core.display import Video
import cv2  # Biblioteca para manipulação de vídeo

# Inicialização do detector
detector = Detector()
fex = Fex()

# Caminho para o vídeo 
test_video_path = r"C:\Users\Asus\Downloads\video_teste.mp4"

# Exibir o vídeo
display(Video(test_video_path, embed=True))

# Detectar faces no vídeo, skip_frames melhora desempenho
video_prediction = detector.detect_video(
    test_video_path, data_type="video", skip_frames=20, face_detection_threshold=0.95
)

# Exibir as primeiras linhas da predição
print(video_prediction.head())
print(video_prediction.shape)
print(video_prediction.identities)

# Plotar as emoções ao longo do vídeo
plt.figure(figsize=(15, 10))
axes = video_prediction.emotions.plot(title="Emotions throughout the video")
plt.show()
plt.close("all")

# Visualizar detecções no vídeo
figs = video_prediction.plot_detections(poses=True)
plt.show()

# Abrir o vídeo com OpenCV
cap = cv2.VideoCapture(test_video_path)

try:
    # Verificar se o vídeo foi aberto corretamente
    if not cap.isOpened():
        raise ValueError(f"Não foi possível abrir o vídeo: {test_video_path}")

    # Obter o número total de frames do vídeo
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        raise ValueError("O vídeo não contém frames válidos.")

    # Processar frames
    for emotion in video_prediction.emotions.columns:
        filtered_frames = video_prediction[video_prediction.emotions[emotion] > 0.8]

        if not filtered_frames.empty:
            for i, frame_number in enumerate(filtered_frames['frame']):
                if i >= 5:  # Limitar o número de frames processados
                    break

                if frame_number >= total_frames or frame_number < 0:
                    print(f"Frame {frame_number} está fora do alcance do vídeo.")
                    continue

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                if not ret:
                    print(f"Não foi possível capturar o frame {frame_number}.")
                    continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                plt.figure(figsize=(20, 16))
                plt.imshow(frame_rgb)
                plt.title(f"Frame {frame_number} - {emotion} > 0.8")
                plt.axis('off')
                plt.pause(0.001)
                plt.close('all')
        else:
            print(f"Nenhum frame com {emotion} > 0.8.")
finally:
    cap.release()
    print("Recurso de vídeo liberado.")