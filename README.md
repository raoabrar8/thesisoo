# Anomaly Detection System (UCF-Crime)

This project implements an end-to-end anomaly detection system for surveillance videos using the [UCF-Crime dataset](https://www.crcv.ucf.edu/projects/ucf-crime/).  
It includes:
- Preprocessing and frame extraction
- Feature extraction (MobileNetV2 + Optical Flow)
- Feature combination
- Self-supervised LSTM model training
- Anomaly detection and evaluation
- A full-featured Tkinter GUI for interactive analysis and visualization

---

## 📂 Project Structure

```
.
├── main.py                  # One-click pipeline runner
├── requirements.txt
├── data/
│   └── UCF-Crime/           # Put your videos here (organized by class)
├── models/                  # Trained models saved here
├── results/                 # Analysis results and plots
└── src/
    ├── preprocessing/
    ├── features/
    ├── inference/
    ├── selfsupervised/
    ├── interface/
    └── utils/
```

---

## 🚀 Quick Start

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Download and organize the UCF-Crime videos**

   Place videos in `data/UCF-Crime/<ClassName>/*.mp4`  
   Example:  
   ```
   data/UCF-Crime/Abuse/Abuse001_x264.mp4
   data/UCF-Crime/Normal/Normal_Videos_015_x264.mp4
   ```

3. **Run the full pipeline**

   ```bash
   python main.py
   ```

   This will:
   - Extract frames
   - Extract features
   - Train the LSTM anomaly detector
   - Evaluate results
   - Launch the interactive GUI

---

## 🖥️ GUI Usage

- Browse and analyze any video.
- View anomaly scores per frame.
- Play video with anomaly highlights.
- Batch analysis, export results, and real-time webcam support.

![GUI Screenshot](docs/gui_screenshot.png) <!-- Replace with your actual screenshot path -->

---

## 🧩 Main Components

- **Preprocessing**:  
  `src/preprocessing/extract_and_preprocess_frames.py`  
  Extracts & normalizes frames from videos.

- **Feature Extraction**:  
  - `src/features/extract_mobilenet_features.py`
  - `src/features/extract_optical_flow.py`
  - `src/features/combine_features.py`

- **Model Training**:  
  - `src/selfsupervised/train_lstm_predictor_combined.py`  
    Trains LSTM on combined features.

- **Inference & Evaluation**:  
  - `src/inference/evaluate_anomaly_detection_combined.py`

- **GUI**:  
  - `src/interface/tk_anomaly_gui.py`

---

## ⚙️ Customization

- **Threshold**:  
  Set the anomaly threshold in the GUI or in `main.py`.
- **Supported Formats**:  
  Default is `.mp4`. To add others, edit the file search in `main.py`.
- **Batch Processing**:  
  Use the “Batch Analyze” button in the GUI for multi-video processing.

---

## 📝 Citation

If you use this code or system in your research, please cite:

```
@article{ucfcrime,
  title={UCF-Crime: A New Large-Scale Dataset for Anomaly Detection in Surveillance Videos},
  author={Waqas Sultani, Chen Chen, Mubarak Shah},
  journal={arXiv preprint arXiv:1801.04264},
  year={2018}
}
```

---

## 🙏 Acknowledgments

- [UCF-Crime Dataset](https://www.crcv.ucf.edu/projects/ucf-crime/)
- [TensorFlow](https://www.tensorflow.org/)
- [OpenCV](https://opencv.org/)
- [Tkinter](https://docs.python.org/3/library/tkinter.html)

---

## 📧 Contact

For questions or issues, please open an [Issue](https://github.com/yourusername/yourrepo/issues) or contact [your.email@domain.com](mailto:your.email@domain.com).
