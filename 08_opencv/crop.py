"""
图像的裁剪：随机裁剪、中心裁剪
"""
import cv2 as cv
import numpy as np


# 随机裁剪
def random_crop(img, cw, ch):
    h, w = img.shape[:2]
    start_x = np.random.randint(0, w - cw)
    start_y = np.random.randint(0, h - ch)

    # 切片
    return img[start_y:start_y + ch, start_x:start_x + cw]  # 不写最后一个：，兼容三维和二维图片


# 中心裁剪
def center_crop(img, cw, ch):
    h, w = img.shape[:2]
    start_x = (w - cw) // 2
    start_y = (h - ch) // 2

    # 切片
    return img[start_y:start_y + ch, start_x:start_x + cw]  # 不写最后一个：，兼容三维和二维图片


if __name__ == '__main__':
    img = cv.imread("../dl_data/banana_1.png")
    cv.imshow("img", img)

    img_random_crop = random_crop(img, 200, 200)
    cv.imshow("img_random_crop", img_random_crop)

    img_center_crop = center_crop(img, 200, 200)
    cv.imshow("img_center_crop", img_center_crop)

    cv.waitKey()
    cv.destroyAllWindows()
