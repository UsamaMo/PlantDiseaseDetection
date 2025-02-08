import cv2
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

# Download the YOLO model
repo_id = "peachfawn/yolov8-plant-disease"
model_filename = "best.pt"
model_path = hf_hub_download(repo_id=repo_id, filename=model_filename)
print(model_path)
model = YOLO(model_path)

# # Start the webcam capture
# cap = cv2.VideoCapture(0)  # 0 is usually the default webcam
# if not cap.isOpened():
#     print("Error: Could not access the camera.")
#     exit()

# while True:
#     # Read a frame from the webcam
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: Failed to capture image.")
#         break
    
#     # Resize frame to 640x640 (YOLO's default input size)
#     frame_resized = cv2.resize(frame, (640, 640))

#     # Perform detection on the resized frame (convert to RGB if necessary)
#     results = model(frame_resized)

#     # Show the results on the frame
#     frame_with_results = results[0].plot()  # Draws the bounding boxes and labels on the frame

#     # Display the resulting frame
#     cv2.imshow("Live Plant Disease Detection", frame_with_results)

#     # Exit on 'q' key press
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the webcam and close any OpenCV windows
# cap.release()
# cv2.destroyAllWindows()
