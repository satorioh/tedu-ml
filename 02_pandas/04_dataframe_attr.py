import pandas as pd

list01 = [['Alex', 10], ['Bob', 12], ['Clarke', 13]]
df = pd.DataFrame(list01, index=['r1', 'r2', 'r3'], columns=['Name', 'age'])
print(df.axes)  # [Index(['r1', 'r2', 'r3'], dtype='object'), Index(['Name', 'age'], dtype='object')]
print(df.index)  # Index(['r1', 'r2', 'r3'], dtype='object')
print(df.columns)  # Index(['Name', 'age'], dtype='object')
print(df.empty)  # False
print(df.ndim)  # 2
print(df.shape)  # (3, 2)
print(df.size)  # 6
print(df.dtypes)  # 返回Series
"""
Name    object
age      int64
"""
print(df.values)  # 返回ndarray [['Alex' 10]['Bob' 12]['Clarke' 13]]
print(df.head(2))
"""
    Name  age
r1  Alex   10
r2   Bob   12
"""
print(df.tail(2))
"""
      Name  age
r2     Bob   12
r3  Clarke   13
"""
