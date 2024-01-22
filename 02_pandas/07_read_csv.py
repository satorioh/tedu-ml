import pandas as pd

data = pd.read_csv('../data_test/aapl.csv', header=None,
                   names=['name', 'date', '_', 'open', 'high', 'low', 'close', '__'],
                   usecols=['open', 'high', 'low', 'close'])
print(data)

# 写入
data.to_csv('../dist/new_aapl.csv')
