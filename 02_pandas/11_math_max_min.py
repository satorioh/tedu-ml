import numpy as np
import pandas as pd

# 产生9个介于[10, 100)区间的随机数
a = np.random.randint(10, 100, 9)
print(a)
print(np.max(a), np.min(a), np.ptp(a))
print("==" * 20)

data = pd.read_json("../data_test/ratings.json")
fracture = data.loc['Fracture']
# 使用numpy获取
max_index = np.argmax(fracture)
max_score = np.max(fracture)
print(f"{max_index}给fracture这部电影打了最高分：{max_score}")  # 1给fracture这部电影打了最高分：5.0

# 使用pandas获取
max_label = fracture.idxmax()
pd_max_score = fracture.max()
print(f"{max_label}给fracture这部电影打了最高分：{pd_max_score}")  # Michelle Peterson给fracture这部电影打了最高分：5.0
