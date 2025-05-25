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

# Path
data_dir = os.path.join(os.path.dirname(__file__), "input_data")

# Check if the folder exists
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"'input_data' was not {data_dir}")

# Relative path to the template_config.json file
config_path = os.path.join(os.path.dirname(__file__), "template_config.json")
if not os.path.exists(config_path):
    raise FileNotFoundError(f"'template_config.json' was not found in {config_path}")

# Load settings from template_config.json file
with open(config_path, "r") as config_file:
    config = json.load(config_file)

# Determine if the script should process images, videos, or both
process_types = config["data_processing"].get("process_type", ["image", "video"])

# Get all files in the input_data folder
input_files = glob(os.path.join(data_dir, "*"))

# Process images
if "image" in process_types:
    image_files = [f for f in input_files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    for image_path in image_files:
        print(f"Processing images: {image_path}")
        # Show the image
        imshow(image_path)

        # Detect features in the image
        prediction = detector.detect_image(image_path, data_type="image")

        # Show results
        print(prediction.head())
        print(prediction.aus)  # Action Units
        print(prediction.emotions)  # Emotions
        print(prediction.poses)  # Head pose
        print(prediction.identities)  # Identities

        # Plot detections with poses
        figs = prediction.plot_detections(poses=True)
        plt.show()

        # Save the output to a CSV file
        output_csv_path = os.path.join(data_dir, "output.csv")
        prediction.to_csv(output_csv_path, index=False)
        print(f"Output CSV saved to {output_csv_path} and added to input data directory.")

# Process videos
if "video" in process_types:
    video_files = [f for f in input_files if f.lower().endswith((".mp4", ".avi", ".mov"))]
    for video_path in video_files:
        print(f"Processing video: {video_path}")

        # Detect features in the video
        video_prediction = detector.detect_video(video_path, data_type="video", skip_frames=20)

        # Add a label column for detected faces
        video_prediction['label'] = video_prediction.index.map(lambda x: f"Face_{x}")

        # Show the first lines of the prediction DataFrame
        print(video_prediction.head())

        # Open the video with OpenCV
        cap = cv2.VideoCapture(video_path)

        # Check if the video was opened correctly
        if not cap.isOpened():
            print(f"Could not open the video: {video_path}")
            continue
       
        # Plot detected emotions
        plt.figure(figsize=(15, 10))
        video_prediction.emotions.plot(title="Detected Emotions")
        plt.xlabel("Frames")
        plt.ylabel("Emotion Intensity")
        plt.legend(loc="upper right")
        plt.show()
        
        # Loop to process emotions
        for emotion in video_prediction.emotions.columns:
            filtered_frames = video_prediction[video_prediction.emotions[emotion] > 0.8]

            if not filtered_frames.empty:
                print(f"Frames with {emotion} > 0.8:")
                print(filtered_frames[['frame', emotion, 'label']])  # Includes the label in the print

                # Show the corresponding frames
                for frame_number in filtered_frames['frame']:
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
                    plt.title(f"Frame {frame_number} - {emotion} > 0.8")
                    plt.axis('off')
                    plt.show()
            else:
                print(f"No frame had {emotion} > 0.8.")

        # Release the video
        cap.release()

        # Save the output to a CSV file
        output_csv_path = os.path.join(data_dir, "video_output.csv")
        video_prediction.to_csv(output_csv_path, index=False)
        #video_prediction = read_feat(data_csv_path)
        print(f"Output CSV saved to {output_csv_path} and added to input data directory.")

# Processing time for image and video
time2 = time.perf_counter()
print(f"Processing time: {time2 - time1} seconds")

