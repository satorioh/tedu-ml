"""
图像的膨胀
"""
import cv2 as cv
import numpy as np

img = cv.imread("../dl_data/9.png")
cv.imshow("img", img)

# 腐蚀
kernel = np.ones((3, 3), np.uint8)
dilate = cv.dilate(img, kernel, iterations=3)
cv.imshow("dilate", dilate)

cv.waitKey(0)
cv.destroyAllWindows()
