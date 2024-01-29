"""
灰度图像的直方图均衡化
"""
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('../dl_data/sunrise.jpg', 0)  # 0表示灰度图像
cv2.imshow('img', img)

# 直方图均衡化
img_eq = cv2.equalizeHist(img)
cv2.imshow('img_eq', img_eq)

# 绘制原始图像的直方图
plt.subplot(2, 1, 1)
plt.hist(img.ravel(), 256, range=(0, 256))  # hist只接受一维数组，ravel()将多维数组转化为一维数组
plt.subplot(2, 1, 2)
plt.hist(img_eq.ravel(), 256, range=(0, 256))
plt.show()

cv2.waitKey()
cv2.destroyAllWindows()
