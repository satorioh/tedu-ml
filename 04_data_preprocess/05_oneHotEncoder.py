"""
独热编码
"""
import numpy as np
import sklearn.preprocessing as sp

raw_samples = np.array([[1, 3, 2],
                        [7, 5, 4],
                        [1, 8, 6],
                        [7, 3, 9]])
encoder = sp.OneHotEncoder(sparse_output=False, dtype="int32")
res = encoder.fit_transform(raw_samples)
print(res)
"""
[[1 0 1 0 0 1 0 0 0]
 [0 1 0 1 0 0 1 0 0]
 [1 0 0 0 1 0 0 1 0]
 [0 1 1 0 0 0 0 0 1]]
"""
print(encoder.inverse_transform(res))  # 解码
"""
[[1 3 2]
 [7 5 4]
 [1 8 6]
 [7 3 9]]
"""
