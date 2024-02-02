"""
感知机的简单实现
"""


# 逻辑与
def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1 * w1 + x2 * w2
    if tmp <= theta:
        return 0
    else:
        return 1


print(AND(0, 0))  # 0
print(AND(1, 0))  # 0
print(AND(0, 1))  # 0
print(AND(1, 1))  # 1
print("==" * 20)


# 逻辑或
def OR(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.3
    tmp = x1 * w1 + x2 * w2
    if tmp <= theta:
        return 0
    else:
        return 1


print(OR(0, 0))  # 0
print(OR(1, 0))  # 1
print(OR(0, 1))  # 1
print(OR(1, 1))  # 1
print("==" * 20)


# 逻辑异或
def XOR(x1, x2):
    s1 = not AND(x1, x2)
    s2 = OR(x1, x2)
    return AND(s1, s2)


print(XOR(0, 0))  # 0
print(XOR(1, 0))  # 1
print(XOR(0, 1))  # 1
print(XOR(1, 1))  # 0
