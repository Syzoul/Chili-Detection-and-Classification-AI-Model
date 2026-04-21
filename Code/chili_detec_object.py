import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
import tensorflow as tf
import matplotlib.pyplot as plt
import time

# ====== CONFIG ======
YOLO_MODEL_PATH = "D:\\My saved documents\\code\\Chilies detection\\YOLO data\\YOLO_model.pt"
CNN_MODEL_PATH = "D:\\My saved documents\\code\\Chilies detection\\chili_cnn_GoogLeNet.h5"
VIDEO_PATH = "D:\\My saved documents\\code\\Chilies detection\\YOLO data\\VID_20260109_171435.mp4"
OUTPUT_PATH = "D:\\My saved documents\\code\\Chilies detection\\VID_result.mp4"

YOLO_CONF = 0.25
CNN_INPUT_SIZE = (224, 224)
YOLO_INTERVAL = 3       # YOLO mỗi 3 frame
CNN_INTERVAL = YOLO_INTERVAL * 5  # CNN mỗi 15 frame (3 lần YOLO)

CLASSES = ['Normal', 'Dry', 'Defective']
COLORS = [(0, 255, 0), (0, 255, 255), (0, 0, 255)]

# ====== GPU SETUP ======
device = 'cuda' if tf.config.list_physical_devices('GPU') else 'cpu'
print(f"Using device: {device}")

# TensorFlow GPU setup
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

# ====== LOAD MODELS ======
yolo = YOLO(YOLO_MODEL_PATH)
cnn = load_model(CNN_MODEL_PATH)

# ====== VIDEO ======
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

# Matplotlib setup
plt.ion()
fig, ax = plt.subplots(figsize=(10, 6))

frame_count = 0
boxes = []
labels = []

start_time = time.time()

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # ===== YOLO mỗi 3 frame =====
        if frame_count % YOLO_INTERVAL == 0:
            results = yolo(frame, conf=YOLO_CONF, verbose=False)[0]
            boxes = []
            if results.boxes:
                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    boxes.append((x1, y1, x2, y2))
        
        # ===== CNN mỗi 15 frame =====
        if frame_count % CNN_INTERVAL == 0 and boxes:
            crops = []
            for x1, y1, x2, y2 in boxes:
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                h, w, _ = crop.shape
                # Padding trắng để vuông
                if h > w:
                    pad = (h - w) // 2
                    crop_square = cv2.copyMakeBorder(crop, 0, 0, pad, h - w - pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))
                elif w > h:
                    pad = (w - h) // 2
                    crop_square = cv2.copyMakeBorder(crop, pad, w - h - pad, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
                else:
                    crop_square = crop
                crop_resized = cv2.resize(crop_square, CNN_INPUT_SIZE) / 255.0
                crops.append(crop_resized)
            
            if crops:
                preds = cnn.predict(np.array(crops), verbose=0)
                labels = []
                for pred in preds:
                    idx = np.argmax(pred)
                    labels.append((CLASSES[idx], pred[idx]))
        
        # ===== Vẽ bounding box + label =====
        display = frame.copy()
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            if i < len(labels):
                cls, conf = labels[i]
                color_idx = CLASSES.index(cls)
                label_text = f"{cls} {conf*100:.1f}%"
            else:
                color_idx = 0
                label_text = "Processing..."
            cv2.rectangle(display, (x1, y1), (x2, y2), COLORS[color_idx], 2)
            cv2.putText(display, label_text, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS[color_idx], 2)
        
        # ===== Hiển thị video =====
        ax.clear()
        ax.imshow(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
        ax.set_title(f'Frame {frame_count}')
        ax.axis('off')
        plt.pause(1/fps)
        
        # ===== Lưu video =====
        out.write(display)

except KeyboardInterrupt:
    print("Stopped by user")

finally:
    cap.release()
    out.release()
    plt.ioff()
    total_time = time.time() - start_time
    print(f"Processed {frame_count} frames in {total_time:.2f}s (~{frame_count/total_time:.1f} FPS)")
    print(f"Saved result video to: {OUTPUT_PATH}")
