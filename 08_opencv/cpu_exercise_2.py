"""
cpu瑕疵检测
"""
import cv2 as cv
import numpy as np

# 读取图片
img = cv.imread('../dl_data/CPU3.png')

# 灰度化
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('img_gray', img_gray)

# 二值化
t, binary = cv.threshold(img_gray, 166, 255, cv.THRESH_BINARY)
cv.imshow('binary', binary)

# 查找轮廓
contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
print("轮廓数量", len(contours))

# 绘制轮廓
binary_fill_img = binary.copy()
cv.drawContours(binary_fill_img, contours, -1, (255, 255, 255), -1)
cv.imshow('contours', binary_fill_img)

# 用填充后的图片减去二值化图片
binary_sub_img = cv.subtract(binary_fill_img, binary)
cv.imshow('binary_sub_img', binary_sub_img)

# 闭运算
kernel = np.ones((3, 3), np.uint8)
close = cv.morphologyEx(binary_sub_img, cv.MORPH_CLOSE, kernel)
cv.imshow('close', close)

# 高斯模糊
gaussian = cv.GaussianBlur(close, (3, 3), 0)
cv.imshow('gaussian', gaussian)

# 查找轮廓
contours, hierarchy = cv.findContours(gaussian, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# 绘制轮廓
img_copy = img.copy()
cv.drawContours(img_copy, contours, -1, (0, 0, 255), 2)
cv.imshow('img_copy', img_copy)

cv.waitKey(0)
cv.destroyAllWindows()
