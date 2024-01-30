"""
图像相加：尺寸需要一致
"""
import cv2 as cv

cat = cv.imread("../dl_data/cat_avatar.png", 1)
fu = cv.imread("../dl_data/fu.jpg", 1)
fu_resize = cv.resize(fu, (cat.shape[1], cat.shape[0]))
cv.imshow("cat", cat)
cv.imshow("fu", fu_resize)

# 图像直接相加
# add_img = cv.add(lena, lily)  # 图像直接相加，会导致图像过亮、过白
# cv.imshow("add_img", add_img)
#
# 图像权重相加
add_weighted_img = cv.addWeighted(cat, 0.2, fu_resize, 0.8, 20)  # 亮度调节量为50
cv.imshow("add_weighted_img", add_weighted_img)
cv.imwrite("../dist/fu_cat.jpg", add_weighted_img)

cv.waitKey()
cv.destroyAllWindows()
