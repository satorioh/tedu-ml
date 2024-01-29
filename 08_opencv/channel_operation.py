"""
对彩色图像的某个通道进行操作
"""
import cv2

img = cv2.imread("../dl_data/opencv2.png")
print(img)
cv2.imshow("img", img)

# 只取蓝色通道(分量法取蓝色)
blue_only = img[:, :, 0]
cv2.imshow("blue_only", blue_only)

# 将原始图像中的蓝色通道，赋值为0（去掉蓝色通道）
img[:, :, 0] = 0
cv2.imshow("img-b0", img)

# 在蓝色为0的基础上，再将绿色赋值为0
img[:, :, 1] = 0
cv2.imshow("img-b0-g0", img)

cv2.waitKey()
cv2.destroyAllWindows()
