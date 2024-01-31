"""
纸张检测
"""
import cv2 as cv

# 读取图片
img = cv.imread('../dl_data/paper.jpg')

# 灰度化
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('img_gray', img_gray)

# 反二值化
# t, binary = cv.threshold(img_gray, 190, 255, cv.THRESH_BINARY)
# cv.imshow('binary', binary)

# 高斯模糊
img_gaussian_blur = cv.GaussianBlur(img, (3, 3), 3)
cv.imshow('img_gaussian_blur', img_gaussian_blur)

# 边缘检测
img_canny = cv.Canny(img_gaussian_blur, 50, 150)
cv.imshow('img_canny', img_canny)

cv.waitKey(0)
cv.destroyAllWindows()
