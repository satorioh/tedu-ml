from ultralytics import YOLO

# Load a model
model = YOLO('../model/yolov8n.pt')  # load a pretrained model (recommended for training)

# Train the model with 2 GPUs
results = model.train(data='./dataset/unimib2016.yaml', epochs=5, imgsz=640, device='mps')
