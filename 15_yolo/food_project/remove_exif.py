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
path = os.path.join(os.getcwd(), './dataset/images')


def rectify_imgs():
    for img_name in os.listdir(path):
        if not img_name[-4:] == img_type:
            continue
        img_path = os.path.join(path, img_name)
        image = Image.open(img_path)
        # img_rectified = Image.fromarray(np.asarray(img))  # 将 img 转化为 ndarray，再将该ndarray转化为Image并保存
        data = list(image.getdata())
        image_without_exif = Image.new(image.mode, image.size)
        image_without_exif.putdata(data)
        image_without_exif.save(img_path)
        print(img_name)


if __name__ == '__main__':
    rectify_imgs()
