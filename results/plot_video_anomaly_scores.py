import numpy as np
import matplotlib.pyplot as plt
import os

# --- CONFIG ---
# Change this to point to a specific video of interest
CLASS = 'Arrest'        # e.g. 'Arrest', 'Assault', etc.
VIDEO = 'Arrest001_x264'  # Name of the video folder (without extension)
FEATURE_PATH = f'data/features_mobilenet/{CLASS}/{VIDEO}/mobilenet_features.npy'
MODEL_PATH = 'models/lstm_predictor.h5'
SEQ_LEN = 10

# --- Load model and features ---
import tensorflow as tf
model = tf.keras.models.load_model(MODEL_PATH)
features = np.load(FEATURE_PATH)

# --- Compute anomaly scores ---
scores = []
for i in range(len(features) - SEQ_LEN):
    inp_seq = features[i:i+SEQ_LEN]
    inp_seq = np.expand_dims(inp_seq, axis=0)
    pred = model.predict(inp_seq, verbose=0)[0]
    true = features[i+SEQ_LEN]
    mse = np.mean((pred - true) ** 2)
    scores.append(mse)
scores = np.array(scores)

# --- Optionally, load true labels for plotting ---
# If you have frame-level ground truth, load and plot here (else skip)
# Example: true_labels = np.zeros_like(scores) # or load from file

plt.figure(figsize=(12,4))
plt.plot(scores, label='Anomaly Score')
# plt.plot(true_labels, label='True Label', alpha=0.5)  # Uncomment if available
plt.title(f'Anomaly Scores for {CLASS}/{VIDEO}')
plt.xlabel('Frame Index')
plt.ylabel('Anomaly Score')
plt.legend()
plt.tight_layout()
plt.show()