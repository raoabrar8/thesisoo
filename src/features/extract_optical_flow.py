import os
import numpy as np
import cv2
from tqdm import tqdm

FRAME_DIR = 'data/processed_frames'
FLOW_DIR = 'data/features_opticalflow'

def compute_dense_optical_flow(video_folder, flow_folder):
    frames = sorted([f for f in os.listdir(video_folder) if f.endswith('.npy')])
    if len(frames) < 2:
        return
    flows = []
    prev = np.load(os.path.join(video_folder, frames[0]))
    prev_gray = cv2.cvtColor(((prev + 1) * 127.5).astype(np.uint8), cv2.COLOR_RGB2GRAY)  # Undo MobileNetV2 normalization

    for i in range(1, len(frames)):
        curr = np.load(os.path.join(video_folder, frames[i]))
        curr_gray = cv2.cvtColor(((curr + 1) * 127.5).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None, 
            pyr_scale=0.5, levels=3, winsize=15, 
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        flows.append(flow)
        prev_gray = curr_gray
    flows = np.array(flows)
    np.save(os.path.join(flow_folder, 'farneback_flow.npy'), flows)

def main():
    os.makedirs(FLOW_DIR, exist_ok=True)
    class_folders = [f for f in os.listdir(FRAME_DIR) if os.path.isdir(os.path.join(FRAME_DIR, f))]
    for class_folder in tqdm(class_folders, desc="Classes"):
        class_path = os.path.join(FRAME_DIR, class_folder)
        out_class_folder = os.path.join(FLOW_DIR, class_folder)
        os.makedirs(out_class_folder, exist_ok=True)
        videos = [v for v in os.listdir(class_path) if os.path.isdir(os.path.join(class_path, v))]
        for video_folder in tqdm(videos, desc=f"Processing {class_folder}", leave=False):
            vid_path = os.path.join(class_path, video_folder)
            out_vid_folder = os.path.join(out_class_folder, video_folder)
            os.makedirs(out_vid_folder, exist_ok=True)
            compute_dense_optical_flow(vid_path, out_vid_folder)

if __name__ == "__main__":
    main()