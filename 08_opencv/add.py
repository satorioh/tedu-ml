"""
图像相加：尺寸需要一致
"""
import cv2 as cv

lena = cv.imread("../dl_data/lena.jpg", 0)
lily = cv.imread("../dl_data/lily_square.png", 0)
cv.imshow("lena", lena)
cv.imshow("lily", lily)

# 图像直接相加
add_img = cv.add(lena, lily)  # 图像直接相加，会导致图像过亮、过白
cv.imshow("add_img", add_img)

# 图像权重相加
add_weighted_img = cv.addWeighted(lena, 0.8, lily, 0.2, 0)  # 亮度调节量为50
cv.imshow("add_weighted_img", add_weighted_img)

cv.waitKey()
cv.destroyAllWindows()
