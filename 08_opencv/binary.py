"""
二值化与反二值化
"""
import cv2

img = cv2.imread("../dl_data/lena.jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("img_gray", img_gray)

# 二值化
t, binary = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY)  # 100为阈值，大于为255，小于为0
cv2.imshow("binary", binary)

# 反二值化
t2, binary_inv = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)  # 100为阈值，大于为0，小于为255
cv2.imshow("binary_inv", binary_inv)

cv2.waitKey(0)
cv2.destroyAllWindows()
