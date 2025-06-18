import os
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses

FEATURE_DIR = '/content/drive/MyDrive/data/features_combined/Normal'
SEQ_LEN = 10
BATCH_SIZE = 32
EPOCHS = 20
MODEL_SAVE_PATH = '/content/drive/MyDrive/models/lstm_predictor_combined.h5'

def load_sequences(feature_dir, seq_len=SEQ_LEN):
    X, y = [], []
    video_folders = [os.path.join(feature_dir, v) for v in os.listdir(feature_dir) if os.path.isdir(os.path.join(feature_dir, v))]
    for vid_folder in tqdm(video_folders, desc="Videos"):
        feat_path = os.path.join(vid_folder, 'combined_features.npy')
        if not os.path.exists(feat_path):
            continue
        feats = np.load(feat_path)  # shape: (num_frames, 1284)
        if feats.shape[0] < seq_len + 1:
            continue
        for i in range(feats.shape[0] - seq_len):
            X.append(feats[i:i+seq_len])
            y.append(feats[i+seq_len])
    X = np.array(X)
    y = np.array(y)
    return X, y

def build_lstm_predictor(seq_len, feature_dim):
    model = models.Sequential([
        layers.Input(shape=(seq_len, feature_dim)),
        layers.LSTM(256, return_sequences=False),
        layers.Dense(feature_dim)
    ])
    model.compile(optimizer=optimizers.Adam(1e-3), loss=losses.MeanSquaredError())
    return model

def main():
    print("Loading sequences...")
    X, y = load_sequences(FEATURE_DIR, SEQ_LEN)
    print(f"Loaded {X.shape[0]} sequences.")

    print("Building model...")
    model = build_lstm_predictor(SEQ_LEN, X.shape[2])

    print("Training...")
    model.fit(X, y, validation_split=0.1, batch_size=BATCH_SIZE, epochs=EPOCHS)

    print(f"Saving model to {MODEL_SAVE_PATH}")
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model.save(MODEL_SAVE_PATH)

if __name__ == "__main__":
    main()
