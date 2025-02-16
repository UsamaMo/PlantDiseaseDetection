import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from sklearn.metrics import confusion_matrix, precision_recall_curve

# Enable interactive mode for Matplotlib
plt.ion()

# Download the YOLO model
repo_id = "peachfawn/yolov8-plant-disease"
model_filename = "best.pt"
model_path = hf_hub_download(repo_id=repo_id, filename=model_filename)
print(model_path)
model = YOLO(model_path)

# Initialize performance tracking
class_counts = defaultdict(int)
fps_values = []
inference_times = []
frame_count = 0
start_time = time.time()
predicted_classes = []
actual_classes = []  # Assuming ground truth labels are available

# Start the webcam capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Resize frame to 640x640
    frame_resized = cv2.resize(frame, (640, 640))

    # Measure inference time
    t1 = time.time()
    results = model(frame_resized)
    t2 = time.time()
    inference_time = t2 - t1
    inference_times.append(inference_time)

    # Track detections per class
    for detection in results[0].boxes.data:
        class_id = int(detection[5].item())
        class_counts[class_id] += 1
        predicted_classes.append(class_id)

    # Calculate FPS
    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time
    fps_values.append(fps)

    # Show detections in the webcam feed
    frame_with_results = results[0].plot()
    cv2.imshow("Live Plant Disease Detection", frame_with_results)

    # ============================ Update Graph Window ============================

    plt.figure("Performance Metrics", figsize=(15, 10))
    plt.clf()  # Clear previous graphs

    # 1️⃣ FPS Over Time
    plt.subplot(2, 2, 1)
    plt.plot(fps_values, label="FPS", color="blue")
    plt.xlabel("Frame Number")
    plt.ylabel("Frames Per Second")
    plt.title("FPS Over Time")
    plt.legend()

    # 2️⃣ Inference Time Per Frame
    plt.subplot(2, 2, 2)
    plt.plot(inference_times, label="Inference Time (s)", color="red")
    plt.xlabel("Frame Number")
    plt.ylabel("Time (s)")
    plt.title("Inference Time Per Frame")
    plt.legend()

    # 3️⃣ Detections Per Class
    plt.subplot(2, 2, 3)
    plt.bar(class_counts.keys(), class_counts.values(), color="green")
    plt.xlabel("Class ID")
    plt.ylabel("Detections")
    plt.title("Detections Per Class")
    plt.xticks(list(class_counts.keys()), labels=[model.names[i] for i in class_counts.keys()], rotation=45)

    # 4️⃣ Confusion Matrix (if available)
    if actual_classes:
        plt.subplot(2, 2, 4)
        conf_matrix = confusion_matrix(actual_classes, predicted_classes)
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=model.names, yticklabels=model.names)
        plt.xlabel("Predicted Class")
        plt.ylabel("Actual Class")
        plt.title("Confusion Matrix")

    # Show graphs in a separate window
    plt.pause(0.001)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
plt.ioff()  # Disable interactive mode after exiting
plt.show()  # Ensure the last update is displayed
