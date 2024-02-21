"""
文本文件读取示例
"""
import tensorflow as tf
import os


def read_csv(filelist):
    # 创建数据集
    dataset = tf.data.TextLineDataset(filelist)
    print("read dataset", dataset)

    # 定义解析函数
    def parse_csv(line):
        return tf.io.decode_csv(line, record_defaults=[['None'], ['None']])

    # 使用map方法应用解析函数
    dataset = dataset.map(parse_csv)
    print("parse dataset", dataset)

    # 随机打乱样本顺序
    dataset = dataset.shuffle(buffer_size=15)  # buffer_size建议设为样本数量，过大会浪费内存空间，过小会导致打乱不充分

    # 批处理
    dataset = dataset.batch(8)  # batch()是使迭代器一次获取多个样本

    return dataset


if __name__ == '__main__':
    # 构建文件列表
    dir_name = './data_test/'
    file_names = os.listdir(dir_name)
    print(filter(lambda x: x.endswith('.csv'), file_names))
    file_list = [os.path.join(dir_name, i) for i in file_names]

    csv_dataset = read_csv(file_list)
    print("csv_dataset", csv_dataset)

    # 创建迭代器
    for x, y in csv_dataset:
        print("x->", x)
        print("y->", y)
"""
    x-> tf.Tensor(
[b'CCCCCCCCCC3' b'BBBBBBBBBB1' b'CCCCCCCCCC5' b'BBBBBBBBBB5'
 b'AAAAAAAAAA2' b'CCCCCCCCCC4' b'AAAAAAAAAA3' b'CCCCCCCCCC2'], shape=(8,), dtype=string)
y-> tf.Tensor([b'C3' b'B1' b'C5' b'B5' b'A2' b'C4' b'A3' b'C2'], shape=(8,), dtype=string)

x-> tf.Tensor(
[b'BBBBBBBBBB4' b'CCCCCCCCCC1' b'AAAAAAAAAA1' b'AAAAAAAAAA4'
 b'AAAAAAAAAA5' b'BBBBBBBBBB2' b'BBBBBBBBBB3'], shape=(7,), dtype=string)
y-> tf.Tensor([b'B4' b'C1' b'A1' b'A4' b'A5' b'B2' b'B3'], shape=(7,), dtype=string)
"""
