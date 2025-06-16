import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

mobilenet_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

def extract_frames(video_path, max_frames=40):  # Lowered to 40 for memory safety
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while cap.isOpened() and count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))
        frames.append(frame)
        count += 1
    cap.release()
    frames = np.array(frames)
    return frames

def extract_mobilenet_features(frames, batch_size=4):  # Small batch size
    features = []
    n = len(frames)
    for i in range(0, n, batch_size):
        batch = frames[i:i+batch_size].astype(np.float32)
        batch_pre = preprocess_input(batch)
        feats = mobilenet_model(batch_pre, training=False).numpy()
        features.append(feats)
    return np.vstack(features)

def extract_optical_flow(frames):
    flows = []
    prev = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    for i in range(1, len(frames)):
        next_ = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev, next_, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flows.append(flow)
        prev = next_
    return np.array(flows)  # shape: (num_frames-1, 224, 224, 2)

def pool_flow_features(flows):
    pooled = []
    for f in flows:
        mean_uv = np.mean(f, axis=(0, 1))     # shape: (2,)
        std_uv = np.std(f, axis=(0, 1))       # shape: (2,)
        pooled.append(np.concatenate([mean_uv, std_uv]))  # shape: (4,)
    return np.array(pooled)

def get_combined_features(video_path):
    frames = extract_frames(video_path, max_frames=40)   # Lowered for safety!
    if len(frames) < 12:  # must have at least seq_len+2 for LSTM
        raise ValueError("Not enough frames in video (need at least 12 for analysis).")
    mobilenet_feats = extract_mobilenet_features(frames)
    flows = extract_optical_flow(frames)
    flow_feats = pool_flow_features(flows)
    min_len = min(len(mobilenet_feats)-1, len(flow_feats))
    combined = np.concatenate([mobilenet_feats[1:1+min_len], flow_feats[:min_len]], axis=1)  # shape: (min_len, 1284)
    return combined