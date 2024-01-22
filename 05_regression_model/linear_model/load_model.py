"""
加载模型，执行预测
"""
import pickle

with open("linear_model.pkl", "rb") as f:
    model = pickle.load(f)

# 预测
pred_y = model.predict([[1.1], [5.1]])
print(pred_y)
