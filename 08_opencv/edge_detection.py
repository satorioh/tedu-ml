"""
锐化：边沿(缘)检测
"""
import cv2 as cv

img = cv.imread("../dl_data/lena.jpg", 0)
cv.imshow("img", img)

# Sobel算子
# cv.CV_64F: 输出图像深度，本来应该设置为-1，但如果设成-1，可能会发生计算错误，所以通常先设置为精度更高的CV_64F
# dx, dy: x和y方向的导数
img_sobel = cv.Sobel(img, cv.CV_64F, 1, 1, ksize=5)
cv.imshow("img_sobel", img_sobel)

# laplacian算子
img_laplacian = cv.Laplacian(img, cv.CV_64F)
cv.imshow("img_laplacian", img_laplacian)

# Canny算子
img_canny = cv.Canny(img, 50, 150)  # 50和150是阈值
cv.imshow("img_canny", img_canny)

cv.waitKey()
cv.destroyAllWindows()
