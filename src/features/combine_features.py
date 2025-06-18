import os
import numpy as np
from tqdm import tqdm

MOBILENET_DIR = '/content/drive/MyDrive/data/features_mobilenet'
FLOW_DIR = '/content/drive/MyDrive/data/features_opticalflow'
COMBINED_DIR = '/content/drive/MyDrive/data/features_combined'

def process_video(mobilenet_path, flow_path, out_path):
    mobilenet_feats = np.load(mobilenet_path)           # shape: (N, 1280)
    flow_feats = np.load(flow_path)                     # shape: (N-1, 224, 224, 2)
    # Pool each flow map to mean and std for both channels (u,v)
    pooled = []
    for f in flow_feats:
        mean_uv = np.mean(f, axis=(0, 1))     # shape: (2,)
        std_uv = np.std(f, axis=(0, 1))       # shape: (2,)
        pooled.append(np.concatenate([mean_uv, std_uv]))  # shape: (4,)
    pooled = np.array(pooled)  # shape: (N-1, 4)
    # For combined features, only keep frames with both mobilenet and flow (frames 1 to N-1)
    min_len = min(mobilenet_feats.shape[0]-1, pooled.shape[0])
    combined = np.concatenate([mobilenet_feats[1:1+min_len], pooled[:min_len]], axis=1)  # shape: (min_len, 1284)
    np.save(out_path, combined)

def main():
    os.makedirs(COMBINED_DIR, exist_ok=True)
    for class_folder in tqdm(os.listdir(MOBILENET_DIR), desc='Classes'):
        class_path = os.path.join(MOBILENET_DIR, class_folder)
        flow_class_path = os.path.join(FLOW_DIR, class_folder)
        out_class_path = os.path.join(COMBINED_DIR, class_folder)
        if not os.path.isdir(class_path): continue
        os.makedirs(out_class_path, exist_ok=True)
        for video_folder in os.listdir(class_path):
            video_path = os.path.join(class_path, video_folder, 'mobilenet_features.npy')
            flow_path = os.path.join(flow_class_path, video_folder, 'farneback_flow.npy')
            out_path = os.path.join(out_class_path, video_folder + '_combined.npy')
            if not os.path.exists(video_path) or not os.path.exists(flow_path):
                continue
            os.makedirs(os.path.join(out_class_path, video_folder), exist_ok=True)
            out_path = os.path.join(out_class_path, video_folder, 'combined_features.npy')
            process_video(video_path, flow_path, out_path)

if __name__ == "__main__":
    main()
