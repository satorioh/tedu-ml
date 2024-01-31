"""
图像模糊：自定义卷积核
"""
import cv2 as cv
import numpy as np

img = cv.imread("../dl_data/salt.jpg")
cv.imshow("img", img)

# 自定义卷积核
kernel = np.ones((5, 5), dtype=np.float32) / 25
# 使用filter2D, 第二个参数为目标图像的所需深度, -1表示和原图像相同
res = cv.filter2D(img, -1, kernel)
cv.imshow("res", res)

cv.waitKey(0)
cv.destroyAllWindows()
