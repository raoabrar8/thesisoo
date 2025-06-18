import ttkbootstrap as tb
from ttkbootstrap.constants import *
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import tensorflow as tf
from feature_extractor import get_combined_features
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import cv2
from PIL import Image, ImageTk
import os
import csv
import time

UCF_CATEGORIES = [
    "Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion", "Fighting",
    "Robbery", "Shooting", "Shoplifting", "Stealing", "Vandalism", "Normal"
]
ANOMALY_MODEL_PATH = 'models/lstm_predictor_combined.h5'
CATEGORY_MODEL_PATH = 'models/ucf_category_classifier.h5'
SEQ_LEN = 10

# --- Load models
def load_model(path):
    return tf.keras.models.load_model(path)

anomaly_model = load_model(ANOMALY_MODEL_PATH)
try:
    category_model = load_model(CATEGORY_MODEL_PATH)
except Exception:
    category_model = None

def classify_ucf_category(features):
    if category_model is None:
        return "Unknown"
    input_feats = features[-SEQ_LEN:].reshape(1, SEQ_LEN, -1)
    pred = category_model.predict(input_feats)
    idx = np.argmax(pred)
    return UCF_CATEGORIES[idx]

def extract_features_from_video(video_path):
    try:
        features = get_combined_features(video_path)
        return features
    except Exception as e:
        messagebox.showerror("Error", f"Feature extraction failed: {e}")
        return None

def predict_anomaly(features, model, threshold):
    scores = []
    for i in range(len(features) - SEQ_LEN):
        inp_seq = features[i:i+SEQ_LEN]
        inp_seq = np.expand_dims(inp_seq, axis=0)
        pred = model.predict(inp_seq, verbose=0)[0]
        true = features[i+SEQ_LEN]
        mse = np.mean((pred - true) ** 2)
        scores.append(mse)
    scores = np.array(scores)
    anomaly_detected = np.any(scores > threshold)
    return anomaly_detected, scores

def select_video():
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi")])
    if file_path:
        video_path_var.set(file_path)

def select_folder():
    folder_path = filedialog.askdirectory()
    if folder_path:
        folder_var.set(folder_path)

def play_video_with_highlight(video_path, scores, threshold):
    cap = cv2.VideoCapture(video_path)
    idx = 0
    player = tk.Toplevel(root)
    player.title("Video Playback (Anomaly Highlight)")
    player.geometry("480x340")
    player.resizable(False, False)
    video_label = tk.Label(player)
    video_label.pack()
    controls = tb.Frame(player)
    controls.pack()
    pause_var = tk.BooleanVar(value=False)
    def toggle_pause():
        pause_var.set(not pause_var.get())
        play_pause_btn.config(text="Resume" if pause_var.get() else "Pause")
    play_pause_btn = tb.Button(controls, text="Pause", command=toggle_pause, bootstyle="info-outline")
    play_pause_btn.pack(side="left", padx=5, pady=5)
    close_btn = tb.Button(controls, text="Close", command=player.destroy, bootstyle="danger-outline")
    close_btn.pack(side="left", padx=5, pady=5)

    def loop():
        idx = 0
        while True:
            if not player.winfo_exists() or not video_label.winfo_exists():
                break
            if pause_var.get():
                player.after(60, loop)
                return
            ret, frame = cap.read()
            if not ret:
                break
            # Only draw highlight if we have a score for this frame
            if idx < len(scores) and scores[idx] > float(threshold):
                cv2.rectangle(frame, (0,0), (frame.shape[1]-1, frame.shape[0]-1), (0,0,255), 8)
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = img.resize((440, 280))
            imgtk = ImageTk.PhotoImage(img)
            if video_label.winfo_exists():
                video_label.imgtk = imgtk
                video_label.config(image=imgtk)
            else:
                break
            idx += 1
            player.update_idletasks()
            player.after(33)
        cap.release()

    threading.Thread(target=loop, daemon=True).start()

def export_csv(video_path, scores, detected, category):
    out_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
    if not out_path:
        return
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Frame Index", "Anomaly Score"])
        for i, score in enumerate(scores):
            writer.writerow([i, score])
        writer.writerow([])
        writer.writerow(["Video", os.path.basename(video_path)])
        writer.writerow(["Anomaly Detected", str(detected)])
        writer.writerow(["Category", category])
    messagebox.showinfo("Export Complete", f"Results exported to:\n{out_path}")

def export_plot():
    out_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Image", "*.png")])
    if not out_path:
        return
    fig = plt.Figure(figsize=(4,1.2), dpi=100)
    a = fig.add_subplot(111)
    a.plot(last_scores, color="#1f6aa5")
    a.set_title("Anomaly Scores per Frame", fontsize=10)
    a.set_xlabel("Frame", fontsize=8)
    a.set_ylabel("Score", fontsize=8)
    a.tick_params(axis='both', labelsize=8)
    a.grid(True, linestyle='--', alpha=0.3)
    fig.savefig(out_path)
    messagebox.showinfo("Export Complete", f"Plot exported to:\n{out_path}")

def analyze():
    video_path = video_path_var.get()
    if not video_path:
        messagebox.showerror("Error", "Please select a valid video file!")
        return
    progress["value"] = 0
    result_var.set("Processing...")
    result_label.configure(bootstyle="warning")
    root.update_idletasks()
    features = extract_features_from_video(video_path)
    if features is None:
        progress["value"] = 0
        return
    progress["value"] = 40
    root.update_idletasks()
    try:
        threshold = float(threshold_var.get())
    except Exception:
        threshold = 0.2
    detected, scores = predict_anomaly(features, anomaly_model, threshold)
    progress["value"] = 80
    root.update_idletasks()
    if detected:
        category = classify_ucf_category(features)
        result_text = f"⚠️  Anomaly Detected!\n\nCategory: {category}"
        result_label.configure(bootstyle="danger")
        icon_label.configure(text="⚠️", bootstyle="danger")
    else:
        result_text = "✅  No anomaly detected.\n\nThis is a normal video."
        result_label.configure(bootstyle="success")
        icon_label.configure(text="✅", bootstyle="success")
        category = "Normal"
    result_var.set(result_text)
    progress["value"] = 100
    # Update plot
    global last_scores, last_detected, last_category, last_video_path
    last_scores = scores
    last_detected = detected
    last_category = category
    last_video_path = video_path
    plot_update(scores)
    # Enable buttons
    play_btn["command"] = lambda: play_video_with_highlight(video_path, scores, threshold)
    play_btn["state"] = "normal"
    export_btn["state"] = "normal"
    export_plot_btn["state"] = "normal"
    leaderboard_add(video_path, detected, category, float(np.max(scores)))
    conf_var.set(f"Max Anomaly Score: {np.max(scores):.4f}")

def plot_update(scores):
    for widget in plot_frame.winfo_children():
        widget.destroy()
    fig = plt.Figure(figsize=(4,1.2), dpi=100)
    a = fig.add_subplot(111)
    a.plot(scores, color="#1f6aa5")
    a.set_title("Anomaly Scores per Frame", fontsize=10)
    a.set_xlabel("Frame", fontsize=8)
    a.set_ylabel("Score", fontsize=8)
    a.tick_params(axis='both', labelsize=8)
    a.grid(True, linestyle='--', alpha=0.3)
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=1, padx=4, pady=4)

def update_threshold(val):
    if last_scores is not None:
        try:
            threshold = float(val)
        except:
            threshold = 0.2
        detected = np.any(last_scores > threshold)
        if detected:
            result_label.configure(bootstyle="danger")
            icon_label.configure(text="⚠️", bootstyle="danger")
        else:
            result_label.configure(bootstyle="success")
            icon_label.configure(text="✅", bootstyle="success")
        plot_update(last_scores)

def batch_analyze():
    folder = folder_var.get()
    if not folder:
        messagebox.showerror("Error", "Please select a folder!")
        return
    filelist = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.mp4', '.avi'))]
    if not filelist:
        messagebox.showinfo("No videos", "No video files found in folder.")
        return
    batch_results = []
    for idx, video in enumerate(filelist):
        status_var.set(f"Processing {os.path.basename(video)} ({idx+1}/{len(filelist)})...")
        root.update_idletasks()
        features = extract_features_from_video(video)
        if features is None:
            continue
        detected, scores = predict_anomaly(features, anomaly_model, float(threshold_var.get()))
        if detected:
            category = classify_ucf_category(features)
        else:
            category = "Normal"
        max_score = float(np.max(scores))
        batch_results.append([os.path.basename(video), detected, category, max_score])
        leaderboard_add(video, detected, category, max_score)
    status_var.set("Batch complete.")
    show_leaderboard(batch_results, batch=True)

def leaderboard_add(video_path, detected, category, max_score):
    filename = os.path.basename(video_path)
    leaderboard.insert("", "end", values=(filename, "Anomaly" if detected else "Normal", category, f"{max_score:.4f}"))

def show_leaderboard(rows=None, batch=False):
    lbwindow = tk.Toplevel(root)
    lbwindow.title("Analysis History" if not batch else "Batch Results")
    tree = ttk.Treeview(lbwindow, columns=("Video", "Anomaly", "Category", "Max Score"), show="headings")
    for col in ("Video", "Anomaly", "Category", "Max Score"):
        tree.heading(col, text=col)
        tree.column(col, width=120)
    tree.pack(fill="both", expand=True, padx=10, pady=10)
    if rows:
        for r in rows:
            tree.insert("", "end", values=r)
    else:
        for item in leaderboard.get_children():
            tree.insert("", "end", values=leaderboard.item(item)['values'])
    export_batch_btn = tb.Button(lbwindow, text="Export Table to CSV", bootstyle="success-outline",
                                 command=lambda: export_leaderboard_csv(rows))
    export_batch_btn.pack(pady=8)

def export_leaderboard_csv(rows):
    out_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
    if not out_path:
        return
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Video", "Anomaly", "Category", "Max Score"])
        for r in rows:
            writer.writerow(r)
    messagebox.showinfo("Export Complete", f"Batch results exported to:\n{out_path}")

def show_about():
    about = tk.Toplevel(root)
    about.title("About")
    about.geometry("420x320")
    about.configure(bg="#eaf6fa")
    try:
        logo = Image.open("logo.png")
        logo = logo.resize((90, 90))
        logo = ImageTk.PhotoImage(logo)
        img_label = tk.Label(about, image=logo, bg="#eaf6fa")
        img_label.image = logo
        img_label.pack(pady=8)
    except Exception:
        pass
    about_text = (
        "Anomaly Detection System (UCF-Crime)\n\n"
        "• Feature extractor: MobileNetV2 (pretrained)\n"
        "• Self-supervised LSTM predictor for anomaly\n"
        "• Evaluated on the UCF-Crime dataset\n\n"
        "Developed by Rao Abrar Nawaz\n"
        "© 2025\n"
    )
    tk.Label(about, text=about_text, bg="#eaf6fa", font=("Segoe UI", 11), justify="center").pack(pady=10)

def real_time_anomaly_detection():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not access webcam!")
        return
    seq = []
    rt_win = tk.Toplevel(root)
    rt_win.title("Real-Time Anomaly Detection (Webcam)")
    rt_win.geometry("540x420")
    rt_win.resizable(False, False)
    video_label = tk.Label(rt_win)
    video_label.pack()
    info_label = tk.Label(rt_win, text="Starting...", font=("Segoe UI", 12), fg="#1f6aa5")
    info_label.pack()

    pause = [False]
    def toggle_pause():
        pause[0] = not pause[0]
        pause_btn.config(text="Resume" if pause[0] else "Pause")
    pause_btn = tb.Button(rt_win, text="Pause", command=toggle_pause, bootstyle="info-outline")
    pause_btn.pack(side="left", padx=10, pady=5)
    close_btn = tb.Button(rt_win, text="Close", command=lambda: [cap.release(), rt_win.destroy()], bootstyle="danger-outline")
    close_btn.pack(side="left", padx=10, pady=5)

    seq = []
    try:
        threshold = float(threshold_var.get())
    except:
        threshold = 0.2
    FRAME_SKIP = 3  # Use every 3rd frame for speed

    mobilenet = tf.keras.applications.MobileNetV2(include_top=False, pooling='avg', weights='imagenet')

    def process():
        nonlocal seq
        frame_idx = 0
        while cap.isOpened() and rt_win.winfo_exists():
            if pause[0]:
                time.sleep(0.1)
                continue
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % FRAME_SKIP != 0:
                continue
            img = cv2.resize(frame, (224, 224))
            img = img[..., ::-1] / 255.0  # BGR to RGB, normalize
            img_batch = np.expand_dims(img, axis=0).astype(np.float32)

            features = mobilenet.predict(img_batch)
            features = features.flatten()
            if features.shape[0] < 1284:
                 features = np.concatenate([features, np.zeros(1284 - features.shape[0])])
            seq.append(features)
            if len(seq) > SEQ_LEN:
                seq = seq[-SEQ_LEN:]

            if len(seq) == SEQ_LEN:
                inp_seq = np.array(seq).reshape(1, SEQ_LEN, -1)
                pred = anomaly_model.predict(inp_seq, verbose=0)[0]
                mse = np.mean((pred - seq[-1]) ** 2)
                if mse > threshold:
                    cv2.rectangle(frame, (0,0), (frame.shape[1]-1, frame.shape[0]-1), (0,0,255), 8)
                    cat = classify_ucf_category(np.array(seq))
                    info = f"⚠️ Anomaly!  Score: {mse:.4f}\nCategory: {cat}"
                    if info_label.winfo_exists():
                        info_label.config(fg="#b22222")
                else:
                    cat = "Normal"
                    info = f"✅ Normal  Score: {mse:.4f}"
                    info_label.config(fg="#228B22")
                info_label.config(text=info)
            else:
                info_label.config(text="Collecting initial frames...", fg="#1f6aa5")

            img_disp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_disp = Image.fromarray(img_disp)
            img_disp = img_disp.resize((500, 350))
            imgtk = ImageTk.PhotoImage(img_disp)
            video_label.imgtk = imgtk
            video_label.config(image=imgtk)
            rt_win.update_idletasks()
            time.sleep(0.03)  # ~30fps

        cap.release()
    threading.Thread(target=process, daemon=True).start()

# --- GUI SETUP ---
root = tb.Window(themename="minty")
root.title("Anomaly Detection System (UCF-Crime)")
root.minsize(900, 600)
root.geometry("990x670")

banner_frame = tb.Frame(root, bootstyle="info")
banner_frame.pack(fill="x", padx=0, pady=0)
try:
    logo = Image.open("logo.png")
    logo = logo.resize((55, 55))
    logo = ImageTk.PhotoImage(logo)
    logo_label = tk.Label(banner_frame, image=logo, bg="#e3f6fa")
    logo_label.pack(side="left", padx=(14,6), pady=(2,2))
except Exception:
    logo_label = tk.Label(banner_frame, text="🎥", font=("Segoe UI", 30), bg="#e3f6fa")
    logo_label.pack(side="left", padx=(18,6), pady=(2,2))
tb.Label(banner_frame, text="Anomaly Detection System (UCF-Crime)", font=("Segoe UI", 22, "bold"),
        bootstyle="info inverse", anchor="center").pack(fill="x", padx=10, pady=(6,4), side="left")
about_btn = tb.Button(banner_frame, text="About", bootstyle="info-outline", command=show_about)
about_btn.pack(side="right", padx=12, pady=16)

main_frame = tb.Frame(root, padding=12, bootstyle="light")
main_frame.pack(padx=14, pady=(2,0), fill="x")

tb.Label(main_frame, text="Select a video file:", font=("Segoe UI", 11)).grid(row=0, column=0, sticky="w", padx=(0,2), pady=2)
video_path_var = tk.StringVar()
tb.Entry(main_frame, textvariable=video_path_var, width=46, font=("Segoe UI", 10)).grid(row=0, column=1, padx=5, pady=2)
tb.Button(main_frame, text="Browse", command=select_video, bootstyle="info-outline").grid(row=0, column=2, padx=5, pady=2)

tb.Label(main_frame, text="Or select folder:", font=("Segoe UI", 11)).grid(row=1, column=0, sticky="w", padx=(0,2), pady=2)
folder_var = tk.StringVar()
tb.Entry(main_frame, textvariable=folder_var, width=46, font=("Segoe UI", 10)).grid(row=1, column=1, padx=5, pady=2)
tb.Button(main_frame, text="Browse", command=select_folder, bootstyle="info-outline").grid(row=1, column=2, padx=5, pady=2)
tb.Button(main_frame, text="Batch Analyze", command=batch_analyze, bootstyle="warning-outline").grid(row=1, column=3, padx=5, pady=2)

tb.Label(main_frame, text="Anomaly threshold:", font=("Segoe UI", 11)).grid(row=2, column=0, sticky="w", padx=(0,2), pady=2)
threshold_var = tk.StringVar(value="0.2")
threshold_entry = tb.Entry(main_frame, textvariable=threshold_var, width=10, font=("Segoe UI", 10))
threshold_entry.grid(row=2, column=1, padx=5, pady=2, sticky="w")
threshold_slider = tb.Scale(main_frame, from_=0.01, to=0.5, orient=tk.HORIZONTAL, length=180, command=lambda val: threshold_var.set(f"{float(val):.3f}"))
threshold_slider.set(0.2)
threshold_slider.grid(row=2, column=2, padx=5, pady=2, sticky="w")

tb.Button(main_frame, text="Analyze Video", command=analyze, bootstyle="primary-outline").grid(row=3, column=1, padx=5, pady=10, sticky="w")
play_btn = tb.Button(main_frame, text="Play Video with Highlight", state="disabled", bootstyle="warning-outline")
play_btn.grid(row=3, column=2, padx=2, pady=10, sticky="w")
export_btn = tb.Button(main_frame, text="Export Results (CSV)", state="disabled", bootstyle="success-outline",
                       command=lambda: export_csv(last_video_path, last_scores, last_detected, last_category))
export_btn.grid(row=3, column=3, padx=2, pady=10, sticky="w")
export_plot_btn = tb.Button(main_frame, text="Export Plot (PNG)", state="disabled", bootstyle="success-outline", command=export_plot)
export_plot_btn.grid(row=3, column=4, padx=2, pady=10, sticky="w")

# --- REAL-TIME BUTTON: Inserted here, after the export_plot_btn and before progress ---
realtime_btn = tb.Button(main_frame, text="Real-Time (Webcam)", bootstyle="danger-outline", command=real_time_anomaly_detection)
realtime_btn.grid(row=3, column=5, padx=2, pady=10, sticky="w")

progress = tb.Progressbar(main_frame, length=200, mode='determinate', bootstyle="info-striped")
progress.grid(row=4, column=1, padx=5, pady=10, sticky="w")

card = tb.Frame(root, bootstyle="secondary", padding=8)
card.pack(fill="x", padx=22, pady=(14,5))
icon_label = tb.Label(card, text="", font=("Segoe UI", 22), bootstyle="secondary", anchor="center", width=2)
icon_label.pack(side="left", padx=(0,8))
result_var = tk.StringVar()
result_label = tb.Label(card, textvariable=result_var, font=("Segoe UI", 13, "bold"), bootstyle="secondary", anchor="w", justify="left")
result_label.pack(side="left", fill="x", expand=True)
conf_var = tk.StringVar()
tk.Label(card, textvariable=conf_var, font=("Segoe UI", 10), bg="#eaf6fa", fg="#1f6aa5").pack(side="right", padx=5)

plot_frame = tb.Frame(root, padding=5, bootstyle="light")
plot_frame.pack(fill="both", expand=True, padx=13, pady=(2,12))

lb_frame = tb.Frame(root, bootstyle="light")
lb_frame.pack(fill="x", padx=20, pady=(0,8))
tb.Label(lb_frame, text="Session History:", font=("Segoe UI", 11, "underline"), bootstyle="secondary").pack(side="left", padx=4)
show_lb_btn = tb.Button(lb_frame, text="Leaderboard", bootstyle="info-outline", command=show_leaderboard)
show_lb_btn.pack(side="left", padx=4)
status_var = tk.StringVar()
tk.Label(lb_frame, textvariable=status_var, font=("Segoe UI", 9), bg="#eaf6fa", fg="#666").pack(side="right", padx=5)
leaderboard = ttk.Treeview(lb_frame, columns=("Video", "Anomaly", "Category", "Max Score"), show="headings", height=2)
for col in ("Video", "Anomaly", "Category", "Max Score"):
    leaderboard.heading(col, text=col)
    leaderboard.column(col, width=110)
leaderboard.pack(fill="x", padx=10, pady=2)

def tooltip(widget, text):
    tip = tk.Toplevel(widget, bg="white")
    tip.withdraw()
    tip.overrideredirect(True)
    label = tk.Label(tip, text=text, bg="white", font=("Segoe UI", 9), relief="solid", borderwidth=1)
    label.pack()
    def enter(event):
        x = event.x_root + 15
        y = event.y_root + 5
        tip.geometry(f"+{x}+{y}")
        tip.deiconify()
    def leave(event):
        tip.withdraw()
    widget.bind("<Enter>", enter)
    widget.bind("<Leave>", leave)
tooltip(threshold_entry, "Type a value or use the slider for anomaly threshold.")
tooltip(threshold_slider, "Adjust and see results instantly.")
tooltip(play_btn, "Play video with anomaly frames outlined.")
tooltip(export_btn, "Export anomaly scores and result to CSV file.")
tooltip(export_plot_btn, "Export anomaly score plot to PNG image.")
tooltip(realtime_btn, "Start live anomaly detection using your webcam.")

last_scores = None
last_detected = None
last_category = None
last_video_path = None

main_frame.columnconfigure(1, weight=1)
main_frame.columnconfigure(0, weight=0)
main_frame.columnconfigure(2, weight=0)
main_frame.columnconfigure(3, weight=0)
main_frame.columnconfigure(4, weight=0)

root.mainloop()
