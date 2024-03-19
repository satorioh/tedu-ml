import cv2
import ultralytics
from ultralytics import YOLO

model = YOLO('./model/yolov8m.pt')
results = model.track(source='./source/Supermarket_Advertisement.mp4', save=True, show=True, persist=True,
                      project='./result', tracker="bytetrack.yaml")
