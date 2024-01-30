"""
开运算：先腐蚀后膨胀
"""
import cv2 as cv
import numpy as np

img = cv.imread("../dl_data/5.png")
cv.imshow("img", img)

# 开运算
kernel = np.ones((3, 3), np.uint8)
res = cv.morphologyEx(img, cv.MORPH_OPEN, kernel, iterations=3)
cv.imshow("res", res)

cv.waitKey(0)
cv.destroyAllWindows()
