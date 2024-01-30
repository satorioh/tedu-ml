"""
形态学梯度：膨胀图像减腐蚀图像
"""
import cv2 as cv
import numpy as np

img = cv.imread("../dl_data/cat_avatar.png")
cv.imshow("img", img)

# 形态学梯度
kernel = np.ones((4, 4), np.uint8)
res = cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel)
cv.imwrite("../dist/cat_morph_gradient.png", res)
cv.imshow("res", res)

cv.waitKey(0)
cv.destroyAllWindows()
