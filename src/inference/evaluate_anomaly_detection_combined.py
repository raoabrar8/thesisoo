import os
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve

MODEL_PATH = 'models/lstm_predictor_combined.h5'
FEATURE_BASE = 'data/features_combined'
SEQ_LEN = 10
SCORES_SAVE = 'results/anomaly_scores_combined.npy'
LABELS_SAVE = 'results/true_labels_combined.npy'
CLASSES = os.listdir(FEATURE_BASE)

def get_label(class_name):
    return 0 if class_name.lower() == "normal" else 1

def get_anomaly_scores_for_video(feature_path, model):
    feats = np.load(feature_path)
    scores = []
    for i in range(len(feats) - SEQ_LEN):
        inp_seq = feats[i:i+SEQ_LEN]
        inp_seq = np.expand_dims(inp_seq, axis=0)
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
            feat_path = os.path.join(class_path, video_folder, 'combined_features.npy')
            if not os.path.exists(feat_path):
                continue
            scores = get_anomaly_scores_for_video(feat_path, model)
            all_scores.extend(scores)
            all_labels.extend([label] * len(scores))
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    np.save(SCORES_SAVE, all_scores)
    np.save(LABELS_SAVE, all_labels)
    auc = roc_auc_score(all_labels, all_scores)
    print(f"AUC-ROC: {auc:.4f}")
    precision, recall, thresholds = precision_recall_curve(all_labels, all_scores)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx]
    preds = (all_scores > best_thresh).astype(int)
    acc = accuracy_score(all_labels, preds)
    print(f"Best threshold: {best_thresh:.5f}")
    print(f"Frame-level Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()