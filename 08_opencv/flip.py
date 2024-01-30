"""
镜像翻转
"""
import cv2 as cv

img = cv.imread("../dl_data/cat_avatar.png")
cv.imshow("img", img)

# 水平翻转
flip_1 = cv.flip(img, 1)
cv.imshow("flip_1", flip_1)

# 垂直翻转
flip_0 = cv.flip(img, 0)
cv.imshow("flip_0", flip_0)

cv.waitKey(0)
cv.destroyAllWindows()
