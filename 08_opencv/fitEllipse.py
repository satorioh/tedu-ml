"""
轮廓拟合：椭圆
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

# 轮廓拟合椭圆
ellipse = cv.fitEllipse(contours[0])

# 绘制轮廓
res = cv.ellipse(img, ellipse, (0, 0, 255), 2)

cv.imshow('res', res)

cv.waitKey()
cv.destroyAllWindows()
