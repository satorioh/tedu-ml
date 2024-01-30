"""
图像的缩放：放大和缩小
"""
import cv2 as cv

img = cv.imread("../dl_data/Linus.png")
cv.imshow("Linus", img)
h, w = img.shape[:2]

# 缩小
dst_size = (int(w / 2), int(h / 2))
small = cv.resize(img, dst_size)
cv.imshow("small", small)

# 放大
dst_size2 = (int(w * 2), int(h * 2))
big_nearest = cv.resize(img, dst_size2, interpolation=cv.INTER_NEAREST)  # 最近邻插值
cv.imshow("big_nearest", big_nearest)
big_linear = cv.resize(img, dst_size2, interpolation=cv.INTER_LINEAR)  # 双线性插值
cv.imshow("big_linear", big_linear)

cv.waitKey(0)
cv.destroyAllWindows()
