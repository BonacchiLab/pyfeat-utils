from feat import Detector
import matplotlib.pyplot as plt
from feat.utils.io import get_test_data_path
from feat.plotting import imshow
import os
from feat import Fex

# Inicializar o detector
detector = Detector()

## Processando uma única imagem ##
# Caminho da imagem
single_face_img_path = r"C:\Users\Asus\Downloads\single_face.jpg"

# Exibir a imagem
imshow(single_face_img_path)

# Criar um objeto Fex
fex = Fex()  # DataFrame especial que armazena e opera na saída do detector

# Primeira análise
single_face_prediction = detector.detect_image(single_face_img_path, data_type="image")

# Exibir resultados
print(single_face_prediction.head())
print(single_face_prediction.aus)
print(single_face_prediction.emotions)
print(single_face_prediction.poses)
print(single_face_prediction.identities)

# Visualizar resultados de detecção
figs = single_face_prediction.plot_detections(poses=True)
print(figs)

# Salvar saída em um arquivo CSV
single_face_prediction.to_csv("output.csv", index=False)

# Ler os dados salvos
from feat.utils.io import read_feat
input_prediction = read_feat("output.csv")

# Visualizar usando o modelo de AU padronizado do Py-Feat
figs1 = single_face_prediction.plot_detections(faces="aus", muscles=True)
print(figs1)

## MULTIPLE FACES ##
multi_face_image_path = os.path.join(get_test_data_path(), "multi_face.jpg")
multi_face_prediction = detector.detect_image(multi_face_image_path, data_type="image")

# Exibir resultados
print(multi_face_prediction)

# Visualizar resultados de detecção
figs2 = multi_face_prediction.plot_detections(add_titles=False)

## MULTIPLE IMAGES ##
img_list = [single_face_img_path, multi_face_image_path]
mixed_prediction = detector.detect_image(img_list, batch_size=1, data_type="image")

# Exibir resultados
print(mixed_prediction)

# Visualizar resultados de detecção
figs3 = mixed_prediction.plot_detections()
print(figs3)