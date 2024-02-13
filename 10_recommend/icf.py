"""
基于物品的协同过滤
"""
import math

import pandas as pd
import numpy as np


class ItemBasedCF:
    def __init__(self, data_file):
        self.data_file = data_file
        self.data = {}  # {用户:{电影1:分数，电影2:分数}}
        self.item_sim = {}  # 电影相似度
        self.read_data()

    def read_data(self):
        for line in open(self.data_file):
            user, movie, score, time = line.strip().split(',')
            self.data.setdefault(user, {})
            self.data[user][movie] = float(score)
        print(self.data)

    # 计算物品相似度
    def item_similarity(self):
        N = {}  # 电影出现的次数
        C = {}  # 电影-电影的共现矩阵：{电影1:{电影2:次数，电影3:次数}}

        for user, movies in self.data.items():
            for m1 in movies:
                N.setdefault(m1, 0)
                N[m1] += 1

                # 共现矩阵
                C.setdefault(m1, {})
                for m2 in movies:
                    if m1 == m2:
                        continue
                    C[m1].setdefault(m2, 0)
                    C[m1][m2] += 1

        # 计算相似度 ：{电影1:{电影2:相似度，电影3:相似度}}
        for m1, related_movies in C.items():
            self.item_sim.setdefault(m1, {})
            for m2, count in related_movies.items():
                self.item_sim[m1][m2] = count / math.sqrt(N[m1] * N[m2])

        return self.item_sim

    # 推荐
    def recommend(self, user, k=3, n=5):
        # 用户看过的电影
        watched_movies = self.data[user]
        # 寻找相似电影
        rank = {}
        for movie, score in watched_movies.items():
            for related_movie, similarity in sorted(self.item_sim[movie].items(), key=lambda x: x[1], reverse=True)[:k]:
                if related_movie in watched_movies:
                    continue
                # 计算推荐值:{电影1:推荐值，电影2:推荐值}
                rank.setdefault(related_movie, 0)
                rank[related_movie] += score * similarity
        # 对推荐值排序
        return dict(sorted(rank.items(), key=lambda x: x[1], reverse=True)[:n])


if __name__ == '__main__':
    # data = pd.read_csv('../data_test/movielens电影数据/ratings.dat', sep='::', header=None, engine='python')
    # data = data.head(1000)
    # data.to_csv('../data_test/movielens电影数据/ratings_1000.csv', index=False, header=False)
    icf = ItemBasedCF('../data_test/movielens电影数据/ratings_1000.csv')
    icf.item_similarity()
    print(icf.recommend('3'))
