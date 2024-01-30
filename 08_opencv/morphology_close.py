"""
闭运算：先膨胀后腐蚀
"""
import cv2 as cv
import numpy as np

img = cv.imread("../dl_data/9.png")
cv.imshow("img", img)

# 闭运算
kernel = np.ones((1, 4), np.uint8)
res = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel, iterations=10)
cv.imshow("res", res)

cv.waitKey(0)
cv.destroyAllWindows()
