import pandas as pd

list01 = [['bill', 30], ['lily', 22], ['tom', 19]]
df = pd.DataFrame(list01, columns=['Name', 'Age'])
print(df)
"""
   Name  Age
0  bill   30
1  lily   22
2   tom   19
"""
df.loc[0, 'Age'] = 666
print(df)

print(df.loc[df['Age'] > 100])
