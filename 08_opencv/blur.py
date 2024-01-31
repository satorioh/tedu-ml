"""
图像模糊：缩小像素与像素之间的差异
"""
import cv2 as cv

img = cv.imread("../dl_data/salt.jpg")
cv.imshow("img", img)

# 均值滤波
img_mean_blur = cv.blur(img, (5, 5))
cv.imshow("img_mean_blur", img_mean_blur)

# 高斯滤波
img_gaussian_blur = cv.GaussianBlur(img, (5, 5), 3)
cv.imshow("img_gaussian_blur", img_gaussian_blur)

# 中值滤波
img_median_blur = cv.medianBlur(img, 5)
cv.imshow("img_median_blur", img_median_blur)

cv.waitKey(0)
cv.destroyAllWindows()
