"""
查找与绘制轮廓
"""
import cv2 as cv
import numpy as np

img = cv.imread('../dl_data/cat_avatar.png')
print(f"image shape: {img.shape}")
cv.imshow('img', img)

# 转换为灰度图
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('gray', gray)

# 把绿色变为白色
# 1. 把图片转为HSV格式
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
# 2. 设定绿色的阈值
lower_green = np.array([20, 30, 35])
upper_green = np.array([77, 255, 255])
# 3. 根据阈值构建掩模
mask = cv.inRange(hsv, lower_green, upper_green)
cv.imshow('mask', mask)
print(f"mask shape: {mask.shape}")

res = cv.addWeighted(gray, 0.7, mask, 0.3, 0)
cv.imshow('res', res)


# 二值化
ret, binary = cv.threshold(res, 90, 255, cv.THRESH_BINARY)
cv.imshow('binary', binary)

# 裁剪图片，去掉边框
h, w = binary.shape
crop_binary = binary[10:h-10, 10:w-10]  # 去掉上下左右各10个像素
cv.imshow('crop_binary', crop_binary)

# 查找轮廓
contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
print('轮廓数量：', len(contours))  # 4
# 对contours按shape[0]进行排序
contours = sorted(contours, key=lambda x: x.shape[0], reverse=True)
for i in range(len(contours)):
    print(f"contour {i} shape: {contours[i].shape}")

# 绘制轮廓
draw = cv.drawContours(img, contours, 0, (0, 0, 255), 2)
cv.imshow('draw', draw)

cv.waitKey()
cv.destroyAllWindows()
