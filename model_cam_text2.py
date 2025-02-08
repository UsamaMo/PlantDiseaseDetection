from ultralytics import YOLO

from huggingface_hub import hf_hub_download


repo_id = "peachfawn/yolov8-plant-disease"
model_filename = "best.pt"

model_path = hf_hub_download(repo_id=repo_id, filename=model_filename)

model = YOLO(model_path)

results = model("tomato-leaves-isolated-on-white-260nw-1251320371.webp")

results[0].show()



