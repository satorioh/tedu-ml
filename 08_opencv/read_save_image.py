"""
读取、显示、保存图像
"""
import cv2

# 读取图像
img = cv2.imread('../dl_data/Linus.png')  # 高216，宽160
print(img.shape)  # (216, 160, 3)： 行（高）、列（宽）、通道数

# 显示图像
cv2.imshow('Linus', img)
cv2.waitKey()  # 进入阻塞状态，等待用户按键反馈
cv2.destroyAllWindows()  # 销毁所有创建的窗口

# 保存图像
cv2.imwrite('../dist/Linus_copy.png', img)
