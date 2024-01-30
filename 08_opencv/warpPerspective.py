"""
透视变换
"""
import cv2 as cv
import numpy as np

img = cv.imread('../dl_data/pers.png')
cv.imshow('img', img)

# 透视变换
h, w = img.shape[:2]
pts1 = np.float32([[58, 2], [167, 9], [8, 196], [126, 196]])  # 输入图像四个顶点坐标
pts2 = np.float32([[16, 2], [167, 8], [8, 196], [169, 196]])  # 输出图像四个顶点坐标
M = cv.getPerspectiveTransform(pts1, pts2)
res = cv.warpPerspective(img, M, (w, h))
cv.imshow('res', res)

cv.waitKey(0)
cv.destroyAllWindows()
