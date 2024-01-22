"""标准差"""
import numpy as np
import pandas as pd

data = pd.read_json("../data_test/ratings.json")
fracture = data.loc['Fracture']
print(fracture.std())  # 0.7637626158259734
print(np.std(fracture, ddof=1))  # 0.7637626158259734
