import cv2
import numpy as np

label = '../source/valid/20151211_122610.txt'
image = '../source/valid/20151211_122610.jpg'

label_list = []
with open(label) as file:
    lines = file.readlines()
    for line in lines:
        line = line.strip().split(' ')
        label_list.append(line)
print(label_list)

# 加载图片
image = cv2.imread(image)

for line in label_list:
    x1, y1, x2, y2, x3, y3, x4, y4 = map(int, line[1:])
    points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    points = points.astype(np.int32)
    cv2.polylines(image, [points], True, (0, 0, 255), 4)

# 显示图片
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
