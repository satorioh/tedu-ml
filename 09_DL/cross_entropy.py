"""
交叉熵
"""
import math

# 假设当前某样本真实类别为3， softmax输出一共有5个类别
y_true = [0, 0, 0, 1, 0]  # 真实概率：0类别概率为0， 1类别概率为0， 2类别概率为0， 3类别概率为1， 4类别概率为0
y1_pred = [0.1, 0.1, 0.1, 0.6, 0.1]  # 预测概率：0类别概率为0.1， 1类别概率为0.1， 2类别概率为0.1， 3类别概率为0.6， 4类别概率为0.1
y2_pred = [0.01, 0.01, 0.05, 0.7, 0.05]  # 3类别概率为0.6
y3_pred = [0.1, 0.03, 0.02, 0.8, 0.05]  # 3类别概率为0.8
y4_pred = [0.02, 0.02, 0.03, 0.9, 0.03]  # 3类别概率为0.9

entropy1 = 0.0
entropy2 = 0.0
entropy3 = 0.0
entropy4 = 0.0

for i in range(len(y_true)):
    entropy1 += y_true[i] * math.log(y1_pred[i])
    entropy2 += y_true[i] * math.log(y2_pred[i])
    entropy3 += y_true[i] * math.log(y3_pred[i])
    entropy4 += y_true[i] * math.log(y4_pred[i])

print("交叉熵1：", -entropy1)
print("交叉熵2：", -entropy2)
print("交叉熵3：", -entropy3)
print("交叉熵4：", -entropy4)

"""
交叉熵1： 0.5108256237659907
交叉熵2： 0.35667494393873245
交叉熵3： 0.2231435513142097
交叉熵4： 0.10536051565782628
"""
