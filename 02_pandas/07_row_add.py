import pandas as pd

df2 = pd.DataFrame([['zs', 12], ['ls', 4]], columns=['Name', 'Age'])
df3 = pd.DataFrame([['ww', 16], ['zl', 8]], columns=['Name', 'Age'])
# 添加行
df4 = pd.concat([df2, df3], ignore_index=True)
print(df4)
"""
  Name  Age
0   zs   12
1   ls    4
2   ww   16
3   zl    8
"""
s = pd.Series(['f', 'm', 'f', 'm'], name='gender')
# 添加列
df5 = pd.concat([df4, s], axis=1)
print(df5)
"""
  Name  Age gender
0   zs   12      f
1   ls    4      m
2   ww   16      f
3   zl    8      m
"""
