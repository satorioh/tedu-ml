"""
纸张: 图像扶正
"""
import cv2 as cv
import numpy as np

# 读取图片
img = cv.imread('../dl_data/paper.jpg')
cv.imshow('paper', img)

# 灰度图
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 二值化
# ret, binary = cv.threshold(gray, 200, 255, cv.THRESH_BINARY)
# cv.imshow('binary', binary)

# 高斯模糊
gaussian = cv.GaussianBlur(gray, (3, 3), 0)
cv.imshow('gaussian', gaussian)

# 边缘检测
canny = cv.Canny(gaussian, 50, 150)
cv.imshow('canny', canny)

# 轮廓检测
contours, hierarchy = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
print("轮廓数量", len(contours))

# 对轮廓按面积排序
contours = sorted(contours, key=lambda x: cv.contourArea(x), reverse=True)
# 找到面积最大的四边形轮廓
for i in contours:
    eps = 0.01 * cv.arcLength(i, True)
    approx = cv.approxPolyDP(i, eps, True)
    if len(approx) == 4:
        paper_points = approx  # 纸张的四个顶点坐标
        break

# 绘制轮廓
draw_img = img.copy()
# 圈出四个顶点：左上角、左下角、右下角、右上角
colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]  # BGR: 红、绿、蓝、青
for index, item in enumerate(paper_points):
    cv.circle(draw_img, tuple(item[0]), 10, colors[index], 2)
cv.imshow('contours', draw_img)

# reshape，透视变换需要二维坐标
print("paper_points shape:", paper_points.shape)  # (4, 1, 2)
paper_points = paper_points.reshape(4, 2).astype(np.float32)
print("paper_points:", paper_points)
"""
[[204.  14.]
 [ 14. 465.]
 [340. 593.]
 [514. 146.]]
"""

# 求出纸张的宽度和高度
h = np.linalg.norm(paper_points[0] - paper_points[1])  # 左上角和左下角的距离
w = np.linalg.norm(paper_points[1] - paper_points[2])  # 左下角和右下角的距离
print(f"paper width: {w}, height: {h}")

# 变换后的四个顶点坐标
dst_points = np.array([[0, 0], [0, h], [w, h], [w, 0]], np.float32)

# 透视变换
M = cv.getPerspectiveTransform(paper_points, dst_points)  # 生成透视变换矩阵
result = cv.warpPerspective(img, M, (int(w), int(h)))
cv.imshow('result', result)

cv.waitKey(0)
cv.destroyAllWindows()
