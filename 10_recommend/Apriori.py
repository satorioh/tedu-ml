'''
关联规则:Apriori
'''


def loadDataSet():
    with open('item.txt', 'r') as f:
        data = f.readlines()
        res = []
        for i in data:
            res.append(i.strip().split(','))
        return res


def createC1(dataSet):
    # 遍历每一条购买纪录的每一个商品
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            # 如果商品存在，不用添加
            if [item] not in C1:
                C1.append([item])
    C1.sort()

    return list(map(frozenset, C1))
    # [{物品},{物品}]


def scanD(D, Ck, min_Support):
    ssCnt = {}  # {frozenset{'bread'}:次数}

    for tid in D:  # [{bread,cake,milk,tea},{},{}]
        for can in Ck:  # [{bread},{milk},{tea}]
            if can.issubset(tid):
                if can not in ssCnt:
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    # print(ssCnt)
    # 计算支持度：  商品组合出现的次数  / 样本总数
    numItems = float(len(D))

    retList = []  # 保存频繁项集
    supportData = {}  # 纪录每一项的支持度
    # 遍历候选集中的每项出现的次数
    for key in ssCnt:
        support = ssCnt[key] / numItems
        # 过滤
        if support >= min_Support:
            retList.insert(0, key)
        supportData[key] = support

    return retList, supportData


def aprioriGen(Lk, k):
    '''
    根据上一个项集的频繁项集，生成下一个项集的候选集
    :param Lk: 上一个项的频繁项集
    :param k: 要生成的项的个数
    :return: 当前项集的候选集
    '''
    # [{'bread','cake},{'cake',milk},{},{},{}]

    retList = []
    for i in range(len(Lk) - 1):
        for j in range(i + 1, len(Lk)):
            if len(Lk[i] & Lk[j]) == (k - 2):
                retList.append(Lk[i] | Lk[j])
    retList = list(set(retList))
    return retList


def apriori(dataSet, min_Support):
    # 整体逻辑的主函数
    # 生成1项集的候选集
    C1 = createC1(dataSet)
    # 将dataSet转成列表套固定集合的形式
    D = list(map(frozenset, dataSet))
    # 过滤最小支持度，得到频繁1项集
    L1, supportData = scanD(D, C1, min_Support)

    L = [L1]  # 存放每一个项集的频繁项集
    k = 2  # 要生成的下一个项集的项的个数

    while len(L[k - 2]) > 0:  # 当前频繁项集中有项，才能生成下一个项集的候选集

        Ck = aprioriGen(L[k - 2], k)
        # 计算支持度，过滤最小支持度
        Lk, supK = scanD(D, Ck, min_Support)
        L.append(Lk)
        supportData.update(supK)
        k += 1

    return L, supportData


if __name__ == '__main__':
    dataSet = loadDataSet()

    L, supk = apriori(dataSet, 0.5)
    # 每一项的频繁项集
    for i in L:
        print(i)

    # for i,j in supk.items():
    #     print(i,':',j)
