"""
cpu瑕疵检测: 直接绘制轮廓
"""
import cv2 as cv

# 读取图片
img = cv.imread('../dl_data/CPU3.png')

# 灰度化
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('img_gray', img_gray)

# 二值化
t, binary = cv.threshold(img_gray, 166, 255, cv.THRESH_BINARY)
cv.imshow('binary', binary)

# 高斯模糊
gaussian = cv.GaussianBlur(binary, (5, 5), 5)
cv.imshow('gaussian', gaussian)

# 边缘检测
canny = cv.Canny(gaussian, 50, 150)
cv.imshow('canny', canny)

# 查找轮廓
contours, hierarchy = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
print("轮廓数量", len(contours))
contours = sorted(contours, key=lambda x: cv.contourArea(x), reverse=True)
for i in contours:
    print(cv.contourArea(i))
# 过滤出面积小于1000的轮廓
contours = [i for i in contours if cv.contourArea(i) < 1000]


# 绘制轮廓
draw_img = img.copy()
cv.drawContours(draw_img, contours, -1, (0, 0, 255), 2)
cv.imshow('contours', draw_img)

cv.waitKey(0)
cv.destroyAllWindows()
