import ultralytics
from ultralytics import settings, YOLO
from PIL import Image

# ultralytics.checks()

# default settings
"""
{
'settings_version': '0.0.4', 
'datasets_dir': '/Users/robin/Git/datasets', 
'weights_dir': '/Users/robin/Git/tedu-ml/weights', 
'runs_dir': '/Users/robin/Git/tedu-ml/runs', 
'uuid': 'd06c1d0236e4750009a90719a5ea81bab898b3a3113ee8c4b222a37ba7fa88f5', 
'sync': True, 
'api_key': '', 
'openai_api_key': '', 
'clearml': True, 
'comet': True, 
'dvc': True, 
'hub': True, 
'mlflow': True, 
'neptune': True, 
'raytune': True, 
'tensorboard': True, 
'wandb': True}
"""

settings.update({'datasets_dir': '/Users/robin/Git/tedu-ml/ultralytics_datasets'})
# View all settings
# print(settings)

# Load a pretrained YOLOv8n model
model = YOLO('yolov8m.pt')

# Define path to the image file
source = './source'
dist = './result'

# Run inference on the source
results = model(source)

for i, r in enumerate(results):
    # Plot results image
    # im_bgr = r.plot()  # BGR-order numpy array
    # im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

    # Show results to screen (in supported environments)
    # r.show()

    # Save results to disk
    r.save(filename=f'{dist}/results{i}.jpg')
