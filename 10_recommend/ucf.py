"""
基于用户的协同过滤
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ratings = pd.read_json("../data_test/ratings.json")
print(ratings)

login_user = 'Michael Henry'

# 使用皮尔逊相关系数计算用户之间的相似度
corr_matrix = ratings.corr()  # 默认method='pearson'
# 画图
# plt.figure(figsize=(10, 8))
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', cbar=True)
# plt.tight_layout()
# plt.show()

corr_scores = corr_matrix[login_user]
# 取强相关的分数
corr_scores = corr_scores[corr_scores > 0.6]
# 去除自己
corr_scores = corr_scores.drop(login_user)

# {'电影1':[[打分1,打分2]，[相似度1,相似度2]]}
rec_movie = {}

for user, score in corr_scores.items():
    # 相似用户都看过哪些电影
    corr_movies = ratings[user].dropna()
    # 判断登录用户是否看过
    for movie, rating in corr_movies.items():
        if np.isnan(ratings[login_user][movie]):
            if movie not in rec_movie:
                rec_movie[movie] = [[rating], [score]]
            else:
                rec_movie[movie][0].append(rating)
                rec_movie[movie][1].append(score)

print(rec_movie)

# 推荐结果
rec_res = {}
for movie, val in rec_movie.items():
    val = np.array(val)
    score = np.dot(val[0], val[1])
    rec_res[movie] = score
rec_res = sorted(rec_res.items(), key=lambda x: x[1], reverse=True)  # 按照分数降序排列
print(rec_res)
