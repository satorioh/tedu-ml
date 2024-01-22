"""
标签编码
"""
import numpy as np
import sklearn.preprocessing as sp

# 转换一维数据
raw_samples = np.array(['audi', 'ford', 'audi',
                        'bmw', 'ford', 'bmw'])
lb_encoder = sp.LabelEncoder()  # 定义标签编码对象
lb_samples = lb_encoder.fit_transform(raw_samples)  # 执行标签编码
print(lb_samples)  # [0 2 0 1 2 1]
print(lb_encoder.inverse_transform(lb_samples))  # 逆向转换 ['audi' 'ford' 'audi' 'bmw' 'ford' 'bmw']
print("==" * 20)

# 转换二维数据
raw_sample_2 = np.array([
    ['audi', 'a2'],
    ['bmw', 'b3'],
    ['ford', 'f4'],
    ['tesla', 't5']
])

encoders = []
result = []
inverse_result = []

for col in raw_sample_2.T:
    encoder = sp.LabelEncoder()
    encoders.append(encoder)
    result.append(encoder.fit_transform(col))
result_array = np.array(result).T
print(result_array)

for index, col in enumerate(result_array.T):
    encoder = encoders[index]
    res = encoder.inverse_transform(col)
    inverse_result.append(res)
print(inverse_result)
"""
[array(['audi', 'bmw', 'ford', 'tesla'], dtype='<U5'), array(['a2', 'b3', 'f4', 't5'], dtype='<U5')]
"""
