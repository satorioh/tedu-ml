"""
彩色图像的直方图均衡化
"""
import cv2

img = cv2.imread("../dl_data/sunrise.jpg")
cv2.imshow("img", img)

# 将彩色图像转换为HSV图像
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# 对图像的亮度进行直方图均衡化
hsv[..., 2] = cv2.equalizeHist(hsv[..., 2])

# 将HSV图像转回BGR图像
equalize = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
cv2.imshow("equalize", equalize)

cv2.waitKey()
cv2.destroyAllWindows()
