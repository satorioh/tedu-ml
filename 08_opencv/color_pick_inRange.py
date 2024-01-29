"""
色彩提取
"""
import cv2
import numpy as np

# 读取图片
img = cv2.imread('../dl_data/opencv2.png')
cv2.imshow('img', img)

# 转换为HSV格式
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 提取蓝色区域
# 蓝色H通道值为120，通常取120上下10的范围
# S通道和V通道通常取50~255间，饱和度太低、色调太暗计算出来的颜色不准确
min_val = np.array([110, 50, 50])
max_val = np.array([130, 255, 255])
# inRange函数提取指定范围内的颜色
mask = cv2.inRange(hsv, min_val, max_val)  # 生成二值图像
cv2.imshow('mask', mask)
blue = cv2.bitwise_and(img, img, mask=mask)
cv2.imshow('blue', blue)

cv2.waitKey(0)
cv2.destroyAllWindows()
