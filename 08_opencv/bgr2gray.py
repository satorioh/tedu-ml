"""
将彩色图像转为灰度图像
"""
import cv2

# cv2中默认读取的图像是BGR格式
img = cv2.imread("../dl_data/lena.jpg")

# 将BGR格式转为灰度图像
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 显示灰度图
cv2.imshow("img_gray", img_gray)
# 显示原图
cv2.imshow("img", img)
cv2.waitKey()
cv2.destroyAllWindows()
