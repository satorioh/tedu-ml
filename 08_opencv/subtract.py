"""
图像相减
"""
import cv2 as cv

src1 = cv.imread("../dl_data/3.png")
src2 = cv.imread("../dl_data/4.png")
cv.imshow("src1", src1)
cv.imshow("src2", src2)

# 相减
subtract_img = cv.subtract(src1, src2)
cv.imshow("subtract_img", subtract_img)

cv.waitKey(0)
cv.destroyAllWindows()
