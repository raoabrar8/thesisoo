import os
import numpy as np
from tqdm import tqdm
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model

# Paths
FRAME_DIR = '/content/drive/MyDrive/data/processed_frames'
FEATURE_DIR = '/content/drive/MyDrive/data/features_mobilenet'
IMG_SIZE = (224, 224, 3)

def load_pretrained_mobilenet():
    base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=IMG_SIZE)
    # Global Average Pooling gives a 1280-dim feature vector per frame
    output_layer = base_model.output
    from tensorflow.keras.layers import GlobalAveragePooling2D
    output_layer = GlobalAveragePooling2D()(output_layer)
    model = Model(inputs=base_model.input, outputs=output_layer)
    return model

def extract_features_for_video(frame_folder, feature_folder, model):
    frames = sorted([f for f in os.listdir(frame_folder) if f.endswith('.npy')])
    features = []
    for frame_file in frames:
        frame_path = os.path.join(frame_folder, frame_file)
        frame = np.load(frame_path)
        frame = np.expand_dims(frame, axis=0)  # Add batch dimension
        feat = model.predict(frame, verbose=0)
        features.append(feat.squeeze())
    features = np.array(features)
    np.save(os.path.join(feature_folder, 'mobilenet_features.npy'), features)

def main():
    os.makedirs(FEATURE_DIR, exist_ok=True)
    model = load_pretrained_mobilenet()
    class_folders = [f for f in os.listdir(FRAME_DIR) if os.path.isdir(os.path.join(FRAME_DIR, f))]
    for class_folder in tqdm(class_folders, desc="Classes"):
        class_path = os.path.join(FRAME_DIR, class_folder)
        out_class_folder = os.path.join(FEATURE_DIR, class_folder)
        os.makedirs(out_class_folder, exist_ok=True)
        videos = [v for v in os.listdir(class_path) if os.path.isdir(os.path.join(class_path, v))]
        for video_folder in tqdm(videos, desc=f"Processing {class_folder}", leave=False):
            vid_path = os.path.join(class_path, video_folder)
            out_vid_folder = os.path.join(out_class_folder, video_folder)
            os.makedirs(out_vid_folder, exist_ok=True)
            extract_features_for_video(vid_path, out_vid_folder, model)

if __name__ == "__main__":
    main()
