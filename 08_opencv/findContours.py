"""
查找与绘制轮廓
"""
import cv2 as cv

img = cv.imread('../dl_data/3.png')
cv.imshow('img', img)

# 转换为灰度图
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 二值化
ret, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
cv.imshow('binary', binary)

# 查找轮廓
contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
print('轮廓数量：', len(contours))  # 4
print(contours[0].shape)  # (257, 1, 2) 第一个轮廓有257个点
print(hierarchy)
"""
[Next, Previous, First_Child, Parent]
[[[ 1 -1 -1 -1]
  [ 2  0 -1 -1]
  [ 3  1 -1 -1]
  [-1  2 -1 -1]]]
"""

# 绘制轮廓
res = cv.drawContours(img, contours, -1, (0, 0, 255), 2)
cv.imshow('res', res)

cv.waitKey()
cv.destroyAllWindows()
