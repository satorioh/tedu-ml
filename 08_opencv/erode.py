"""
图像的腐蚀
"""
import cv2 as cv
import numpy as np

img = cv.imread("../dl_data/5.png")
cv.imshow("img", img)

# 腐蚀
kernel = np.ones((3, 3), np.uint8)
erosion = cv.erode(img, kernel, iterations=3)
cv.imshow("erosion", erosion)

cv.waitKey(0)
cv.destroyAllWindows()
