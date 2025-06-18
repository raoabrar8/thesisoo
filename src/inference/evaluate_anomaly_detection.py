import os
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from sklearn.metrics import roc_auc_score, accuracy_score

# ====== CONFIG ======
MODEL_PATH = 'models/lstm_predictor.h5'
FEATURE_BASE = '/content/drive/MyDrive/data/features_mobilenet'
SEQ_LEN = 10
SCORES_SAVE = 'results/anomaly_scores.npy'
LABELS_SAVE = 'results/true_labels.npy'
CLASSES = os.listdir(FEATURE_BASE)  # All classes/folders

def get_label(class_name):
    # Returns 1 for anomaly, 0 for normal
    return 0 if class_name.lower() == "normal" else 1

def get_anomaly_scores_for_video(feature_path, model):
    feats = np.load(feature_path)  # shape: (num_frames, 1280)
    scores = []
    for i in range(len(feats) - SEQ_LEN):
        inp_seq = feats[i:i+SEQ_LEN]
        inp_seq = np.expand_dims(inp_seq, axis=0)  # shape: (1, SEQ_LEN, 1280)
        pred = model.predict(inp_seq, verbose=0)[0]
        true = feats[i+SEQ_LEN]
        mse = np.mean((pred - true) ** 2)
        scores.append(mse)
    return scores

def main():
    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    all_scores = []
    all_labels = []
    for class_folder in tqdm(CLASSES, desc="Classes"):
        class_path = os.path.join(FEATURE_BASE, class_folder)
        label = get_label(class_folder)
        videos = [v for v in os.listdir(class_path) if os.path.isdir(os.path.join(class_path, v))]
        for video_folder in tqdm(videos, desc=f"Processing {class_folder}", leave=False):
            feat_path = os.path.join(class_path, video_folder, 'mobilenet_features.npy')
            if not os.path.exists(feat_path):
                continue
            scores = get_anomaly_scores_for_video(feat_path, model)
            all_scores.extend(scores)
            all_labels.extend([label] * len(scores))
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    # Save for later threshold tuning/plotting
    os.makedirs(os.path.dirname(SCORES_SAVE), exist_ok=True)
    np.save(SCORES_SAVE, all_scores)
    np.save(LABELS_SAVE, all_labels)
    # AUC-ROC
    auc = roc_auc_score(all_labels, all_scores)
    print(f"AUC-ROC: {auc:.4f}")
    # Accuracy at optimal threshold
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(all_labels, all_scores)
    # Choose threshold that gives best F1
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx]
    preds = (all_scores > best_thresh).astype(int)
    acc = accuracy_score(all_labels, preds)
    print(f"Best threshold: {best_thresh:.5f}")
    print(f"Frame-level Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
