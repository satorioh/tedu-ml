"""
仿射变换：平移、旋转
"""
import cv2 as cv
import numpy as np


# 平移
def translate(img, x, y):
    h, w = img.shape[:2]  # 获取原始图片的高和宽
    # 定义平移矩阵
    M = np.float32([[1, 0, x],
                    [0, 1, y]])
    return cv.warpAffine(img, M, (w, h))


# 旋转
def rotate(img, angle, center=None, scale=1.0):
    h, w = img.shape[:2]

    # 旋转中心默认为图像中心
    if center is None:
        center = (w / 2, h / 2)

    M = cv.getRotationMatrix2D(center, angle, scale)
    return cv.warpAffine(img, M, (w, h))


if __name__ == '__main__':
    img = cv.imread('../dl_data/lena.jpg')
    cv.imshow('img', img)
    # 平移
    img1 = translate(img, 100, 100)
    cv.imshow('translated', img1)

    # 旋转
    img2 = rotate(img, 45)  # 逆时针
    cv.imshow('rotated', img2)

    cv.waitKey(0)
    cv.destroyAllWindows()
