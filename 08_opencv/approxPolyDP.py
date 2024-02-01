"""
轮廓拟合：多边形
"""
import cv2 as cv

img = cv.imread("../dl_data/cloud.png")
cv.imshow("img", img)

# 转换为灰度图
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 二值化
ret, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
cv.imshow('binary', binary)

# 查找轮廓
contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
print('轮廓数量：', len(contours))

# 轮廓拟合多边形
# 高精度
adp1 = img.copy()
eps1 = 0.005 * cv.arcLength(contours[0], True)  # 精度，根据周长计算
points1 = cv.approxPolyDP(contours[0], eps1, True)

# 绘制轮廓1
cv.drawContours(adp1, [points1], 0, (0, 0, 255), 2)
cv.imshow('adp1', adp1)

# 低精度
adp2 = img.copy()
eps2 = 0.01 * cv.arcLength(contours[0], True)  # 精度，根据周长计算
points2 = cv.approxPolyDP(contours[0], eps2, True)

# 绘制轮廓2
cv.drawContours(adp2, [points2], 0, (0, 0, 255), 2)
cv.imshow('adp2', adp2)

cv.waitKey()
cv.destroyAllWindows()
