# UNIMIB2016
# ├── UNIMIB2016-annotations
# │   ├── check_dataset.py
# │   ├── class_count.py
# │   ├── toYolo.py
# │   ├── class_counts_result.csv
# │   └── formatted_annotations
# ├── rectify_imgs.py <--
# ├── labels (1005)
# └── images (1005)

# rectify_imgs.py

import os
from PIL import Image
import numpy as np

# image type
img_type = '.jpg'

# image folder path
path = os.path.join(os.getcwd(), '../source/valid')


def rectify_imgs():
    for img_name in os.listdir(path):
        if not img_name[-4:] == img_type:
            continue
        img_path = os.path.join(path, img_name)
        img = Image.open(img_path)
        img_rectified = Image.fromarray(np.asarray(img))
        img_rectified.save(img_path)
        print(img_name)


if __name__ == '__main__':
    rectify_imgs()
