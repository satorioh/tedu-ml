import os
import shutil
import random

# 获取images文件夹下所有图片的文件名
image_dir = './dataset/images'
image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f)) and f.endswith('.jpg')]
label_dir = './dataset/labels'
label_files = [f for f in os.listdir(label_dir) if os.path.isfile(os.path.join(label_dir, f)) and f.endswith('.txt')]

# 随机打乱文件名
random.shuffle(image_files)

# 按照7:2:1的比例划分图片
total = len(image_files)
train_ratio, val_ratio = 0.7, 0.2
train_files = image_files[:int(total * train_ratio)]
val_files = image_files[int(total * train_ratio):int(total * (train_ratio + val_ratio))]
test_files = image_files[int(total * (train_ratio + val_ratio)):]

# 创建目标文件夹
os.makedirs(f'{image_dir}/train', exist_ok=True)
os.makedirs(f'{image_dir}/val', exist_ok=True)
os.makedirs(f'{image_dir}/test', exist_ok=True)
os.makedirs(f'{label_dir}/train', exist_ok=True)
os.makedirs(f'{label_dir}/val', exist_ok=True)
os.makedirs(f'{label_dir}/test', exist_ok=True)

# 将图片复制到对应的文件夹下
for f in train_files:
    shutil.copy(os.path.join(image_dir, f), f'{image_dir}/train')
    # 找到同名的label_file并复制到对应的文件夹下
    label_file = f.replace('.jpg', '.txt')
    shutil.copy(os.path.join(label_dir, label_file), f'{label_dir}/train')
for f in val_files:
    shutil.copy(os.path.join(image_dir, f), f'{image_dir}/val')
    label_file = f.replace('.jpg', '.txt')
    shutil.copy(os.path.join(label_dir, label_file), f'{label_dir}/val')
for f in test_files:
    shutil.copy(os.path.join(image_dir, f), f'{image_dir}/test')
    label_file = f.replace('.jpg', '.txt')
    shutil.copy(os.path.join(label_dir, label_file), f'{label_dir}/test')
