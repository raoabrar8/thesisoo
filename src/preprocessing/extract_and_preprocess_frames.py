import cv2
import os
import numpy as np
from tqdm import tqdm
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess

# Set your dataset and output paths
VIDEO_DIR = 'data/UCF-Crime'  # Path to UCF-Crime videos (adjust if needed)
OUTPUT_DIR = 'data/processed_frames'  # Where extracted frames will be saved
FPS = 10  # Extract 10 frames per second
IMG_SIZE = (224, 224)

def extract_and_save_frames(video_path, output_folder, fps_desired=FPS):
    cap = cv2.VideoCapture(video_path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if orig_fps == 0:  # Handle corrupted or unreadable videos
        print(f"Warning: {video_path} has FPS=0. Skipping.")
        return
    interval = int(round(orig_fps / fps_desired))
    count, saved = 0, 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            frame = cv2.resize(frame, IMG_SIZE)
            # Convert BGR (OpenCV) to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Normalize using MobileNetV2 preprocessing
            frame = mobilenet_preprocess(np.array(frame, dtype=np.float32))
            # Save as .npy for fast loading later
            frame_name = os.path.join(output_folder, f'frame_{saved:05d}.npy')
            np.save(frame_name, frame)
            saved += 1
        count += 1
    cap.release()

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    class_folders = [f for f in os.listdir(VIDEO_DIR) if os.path.isdir(os.path.join(VIDEO_DIR, f))]
    for class_folder in tqdm(class_folders, desc="Classes"):
        class_path = os.path.join(VIDEO_DIR, class_folder)
        out_class_folder = os.path.join(OUTPUT_DIR, class_folder)
        os.makedirs(out_class_folder, exist_ok=True)
        videos = [v for v in os.listdir(class_path) if v.endswith('.mp4') or v.endswith('.avi')]
        for video in tqdm(videos, desc=f"Processing {class_folder}", leave=False):
            video_path = os.path.join(class_path, video)
            video_name = os.path.splitext(video)[0]
            out_folder = os.path.join(out_class_folder, video_name)
            os.makedirs(out_folder, exist_ok=True)
            extract_and_save_frames(video_path, out_folder, fps_desired=FPS)

if __name__ == "__main__":
    main()