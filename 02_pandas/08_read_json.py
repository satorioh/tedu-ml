import pandas as pd

data = pd.read_json('../data_test/ratings.json')
# print(data)


data2 = {'Name': ['Tom', 'Jack'], 'Age': [28, 34]}
df = pd.DataFrame(data2, index=['s1', 's2'])
print(df.to_json(orient='records'))  # [{"Name":"Tom","Age":28},{"Name":"Jack","Age":34}]
print(df.to_json(orient='index'))  # {"s1":{"Name":"Tom","Age":28},"s2":{"Name":"Jack","Age":34}}
print(df.to_json(orient='columns'))  # {"Name":{"s1":"Tom","s2":"Jack"},"Age":{"s1":28,"s2":34}}
print(df.to_json(orient='values'))  # [["Tom",28],["Jack",34]]
