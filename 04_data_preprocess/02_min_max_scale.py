"""
范围缩放：将每一列的最小值和最大值设为相同的区间:(0,1)
"""
import numpy as np
import sklearn.preprocessing as sp

raw_sample = np.array([[3.0, -100.0, 2000.0],
                       [0.0, 400.0, 3000.0],
                       [1.0, -400.0, 2000.0]])

mms_sample = raw_sample.copy()

# 1.减去最小值
# 2.减完之后的结果/极差
for col in mms_sample.T:
    col_min = col.min()
    col_max = col.max()
    col -= col_min
    col /= (col_max - col_min)

print(mms_sample)
"""
[[1.         0.375      0.        ]
 [0.         1.         1.        ]
 [0.33333333 0.         0.        ]]
"""
print("==" * 20)

# 基于skLearn提供的API实现
scaler = sp.MinMaxScaler()
res = scaler.fit_transform(raw_sample)
print(res)
"""
[[1.         0.375      0.        ]
 [0.         1.         1.        ]
 [0.33333333 0.         0.        ]]
"""
