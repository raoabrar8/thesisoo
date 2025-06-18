import os
import sys
import glob
import subprocess

# ==== CONFIGURATION ====
UCF_VIDEOS_DIR = '/content/drive/MyDrive/data/UCF-Crime'
PROCESSED_FRAMES_DIR = '/content/drive/MyDrive/data/processed_frames'
FEATURES_MOBILENET_DIR = '/content/drive/MyDrive/data/features_mobilenet'
FEATURES_OPTICALFLOW_DIR = '/content/drive/MyDrive/data/features_opticalflow'
FEATURES_COMBINED_DIR = '/content/drive/MyDrive/data/features_combined'
MODELS_DIR = 'models'
RESULTS_DIR = 'results'
LSTM_MODEL_PATH = os.path.join(MODELS_DIR, 'lstm_predictor_combined.h5')

def print_step(msg):
    print(f"\n{'='*10} {msg} {'='*10}")

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def find_videos():
    return glob.glob(os.path.join(UCF_VIDEOS_DIR, "*", "*.mp4"))

def step1_extract_frames():
    print_step("STEP 1: Frame Extraction & Preprocessing")
    for video_path in find_videos():
        class_name = os.path.basename(os.path.dirname(video_path))
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        out_dir = os.path.join(PROCESSED_FRAMES_DIR, class_name, video_name)
        if os.path.exists(out_dir) and len(os.listdir(out_dir)) > 0:
            print(f"[INFO] Frames already extracted for {video_name}")
            continue
        ensure_dir(out_dir)
        print(f"[INFO] Extracting frames for {video_name}...")
        subprocess.run([
            sys.executable, "src/preprocessing/extract_and_preprocess_frames.py",
            "--input", video_path, "--output", out_dir
        ], check=True)

def step2_extract_features():
    print_step("STEP 2: Feature Extraction (MobileNet + OpticalFlow + Combine)")
    for video_path in find_videos():
        class_name = os.path.basename(os.path.dirname(video_path))
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        frames_dir = os.path.join(PROCESSED_FRAMES_DIR, class_name, video_name)
        mobilenet_dir = os.path.join(FEATURES_MOBILENET_DIR, class_name, video_name)
        opticalflow_dir = os.path.join(FEATURES_OPTICALFLOW_DIR, class_name, video_name)
        combined_dir = os.path.join(FEATURES_COMBINED_DIR, class_name, video_name)

        # Mobilenet features
        if not (os.path.exists(mobilenet_dir) and os.listdir(mobilenet_dir)):
            ensure_dir(mobilenet_dir)
            print(f"[INFO] Extracting MobileNet features for {video_name}...")
            subprocess.run([
                sys.executable, "src/features/extract_mobilenet_features.py",
                "--input", frames_dir, "--output", mobilenet_dir
            ], check=True)
        else:
            print(f"[INFO] MobileNet features already extracted for {video_name}")

        # Optical flow features
        if not (os.path.exists(opticalflow_dir) and os.listdir(opticalflow_dir)):
            ensure_dir(opticalflow_dir)
            print(f"[INFO] Extracting Optical Flow features for {video_name}...")
            subprocess.run([
                sys.executable, "src/features/extract_optical_flow.py",
                "--input", frames_dir, "--output", opticalflow_dir
            ], check=True)
        else:
            print(f"[INFO] Optical Flow features already extracted for {video_name}")

        # Combine features
        if not (os.path.exists(combined_dir) and os.listdir(combined_dir)):
            ensure_dir(combined_dir)
            print(f"[INFO] Combining features for {video_name}...")
            subprocess.run([
                sys.executable, "src/features/combine_features.py",
                "--mobilenet", mobilenet_dir,
                "--opticalflow", opticalflow_dir,
                "--output", combined_dir
            ], check=True)
        else:
            print(f"[INFO] Combined features already present for {video_name}")

def step3_train_lstm():
    print_step("STEP 3: Training LSTM Predictor Model")
    if os.path.exists(LSTM_MODEL_PATH):
        print(f"[INFO] Model already trained at {LSTM_MODEL_PATH}")
        return
    subprocess.run([
        sys.executable, "src/selfsupervised/train_lstm_predictor_combined.py",
        "--features_dir", FEATURES_COMBINED_DIR,
        "--output", LSTM_MODEL_PATH
    ], check=True)

def step4_anomaly_detection_evaluation():
    print_step("STEP 4: Anomaly Detection & Evaluation")
    subprocess.run([
        sys.executable, "src/inference/evaluate_anomaly_detection_combined.py",
        "--features_dir", FEATURES_COMBINED_DIR,
        "--model", LSTM_MODEL_PATH,
        "--results_dir", RESULTS_DIR
    ], check=True)

def step5_launch_gui():
    print_step("System is ready to use! Loading GUI for interactive testing...")
    subprocess.Popen([
        sys.executable, "src/interface/tk_anomaly_gui.py"
    ])

if __name__ == "__main__":
    ensure_dir(PROCESSED_FRAMES_DIR)
    ensure_dir(FEATURES_MOBILENET_DIR)
    ensure_dir(FEATURES_OPTICALFLOW_DIR)
    ensure_dir(FEATURES_COMBINED_DIR)
    ensure_dir(MODELS_DIR)
    ensure_dir(RESULTS_DIR)
    step1_extract_frames()
    step2_extract_features()
    step3_train_lstm()
    step4_anomaly_detection_evaluation()
    step5_launch_gui()
