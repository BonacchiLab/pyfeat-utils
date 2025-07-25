# Necessary imports
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

# Start timer
time1 = time.perf_counter()

# PyFeat detector 
detector = Detector()

def get_image_prediction(image_path):
    return detector.detect_image(image_path, data_type="image")

def get_video_prediction(video_path):
    return detector.detect_video(video_path, data_type="video", skip_frames=20)

if __name__ == "__main__":
    # Ask user if they want to visualize the outputs as they are processed
    visualize_outputs = input("Do you want to visualize the outputs as they are processed? (yes/no): ").strip().lower()

    # Relative path to the template_config.json file
    config_path = os.path.join(os.path.dirname(__file__), "template_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"'template_config.json' was not found in {config_path}")

    # Load settings from template_config.json file
    with open(config_path, "r") as config_file:
        config = json.load(config_file)

    # Use the data_pyfeat-utils path from the config file (outside the repo)
    data_dir = os.path.expanduser(config["data_processing"]["data_pyfeat-utils"])

    # Check if the folder exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"'data_pyfeat-utils' was not found at {data_dir}")

    # Determine if the script should process images, videos, or both
    process_types = config["data_processing"].get("process_type", ["image", "video"])

    # Get all files in the data_pyfeat-utils folder
    input_files = glob(os.path.join(data_dir, "*"))

    # Process images
    if "image" in process_types:
        image_files = [f for f in input_files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        for image_path in image_files:
            print(f"Processing images: {image_path}")
            # Show the image
            imshow(image_path)

            # Detect features in the image
            prediction = get_image_prediction(image_path)

            # Show results
            if visualize_outputs == "yes":
                print(prediction.head())
                print("Identities detected:", prediction.identities)
                print(prediction.aus)  # Action Units
                print(prediction.emotions)  # Emotions
                print(prediction.poses)  # Head pose
                # Plot detections with poses and identity in the title
                figs = prediction.plot_detections(poses=True)
                # Se houver múltiplas identities, mostrar no título
                if hasattr(prediction, "identities") and prediction.identities is not None:
                    for i, fig in enumerate(figs if isinstance(figs, list) else [figs]):
                        identity = prediction.identities[i] if i < len(prediction.identities) else "Unknown"
                        fig.suptitle(f"Identity: {identity}")
                        plt.figure(fig.number)
                        plt.show()
                else:
                    plt.show()
            # Salvar o output para um arquivo CSV com a extensão original do arquivo
            base_name = os.path.basename(image_path)
            output_csv_path = os.path.join(data_dir, f"{base_name}.csv")
            prediction.to_csv(output_csv_path, index=False)
            print(f"Output CSV salvo em {output_csv_path} e adicionado ao diretório data_pyfeat-utils.")

    # Process videos
    if "video" in process_types:
        video_files = [f for f in input_files if f.lower().endswith((".mp4", ".avi", ".mov"))]
        for video_path in video_files:
            print(f"Processing video: {video_path}")

            # Detect features in the video
            video_prediction = get_video_prediction(video_path)

            # Add a label column for detected faces
            video_prediction['label'] = video_prediction.index.map(lambda x: f"Face_{x}")

            # Print identities if available
            if hasattr(video_prediction, "identities") and video_prediction.identities is not None:
                print("Identities detected in video:", video_prediction.identities)

            if visualize_outputs == "yes":
                # Open the video with OpenCV
                cap = cv2.VideoCapture(video_path)
                # Check if the video was opened correctly
                if not cap.isOpened():
                    print(f"Could not open the video: {video_path}")
                    continue
                
                # Loop to process emotions
                for emotion in video_prediction.emotions.columns:
                    filtered_frames = video_prediction[video_prediction.emotions[emotion] > 0.8]

                    if not filtered_frames.empty:
                        for idx, row in filtered_frames.iterrows():
                            frame_number = row['frame']
                            label = row['label']
                            identity = None
                            # Try to extract identity index from label if possible
                            if hasattr(video_prediction, "identities") and video_prediction.identities is not None:
                                try:
                                    face_idx = int(str(label).replace("Face_", ""))
                                    identity = video_prediction.identities[face_idx] if face_idx < len(video_prediction.identities) else "Unknown"
                                except Exception:
                                    identity = "Unknown"
                            print(f"Frame {frame_number} with {emotion} > 0.8 for Identity: {identity} (Label: {label})")
                            # Set the video to the specific frame
                            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                            ret, frame = cap.read()
                            if not ret:
                                print(f"Could not read frame {frame_number}.")
                                continue

                            # Convert the frame from BGR to RGB for display
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                            # Show the frame
                            plt.figure(figsize=(10, 6))
                            plt.imshow(frame_rgb)
                            plt.title(f"Frame {frame_number} - {emotion} > 0.8\nIdentity: {identity}")
                            plt.axis('off')
                            plt.show()
                    else:
                        print(f"No frame had {emotion} > 0.8.")

                # Release the video
                cap.release()
            else:
                # If not visualizing, just release the video if it was opened
                if 'cap' in locals() and cap.isOpened():
                    cap.release()

            # Salvar o output para um arquivo CSV com a extensão original do arquivo
            base_name = os.path.basename(video_path)
            output_csv_path = os.path.join(data_dir, f"{base_name}.csv")
            video_prediction.to_csv(output_csv_path, index=False)
            print(f"Output CSV salvo em {output_csv_path} e adicionado ao diretório data_pyfeat-utils.")

    # Processing time for image and video
    time2 = time.perf_counter()
    print(f"Processing time: {time2 - time1} seconds")