'''
图片文件读取示例
'''
import tensorflow as tf
import os


def read_img(filelist):
    # 创建数据集
    dataset = tf.data.Dataset.from_tensor_slices(filelist)

    # 定义解析函数
    def parse_img(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img)
        img_resized = tf.image.resize(img, (250, 250))
        return img_resized

    # 使用map方法应用解析函数
    dataset = dataset.map(parse_img)

    # 批处理
    dataset = dataset.batch(5)

    return dataset


if __name__ == '__main__':
    # 构建文件列表
    dir_name = '../test_img/'
    file_names = os.listdir(dir_name)
    print(file_names)
    file_list = [os.path.join(dir_name, i) for i in file_names]
    print(file_list)

    imgs = read_img(file_list)

    # 创建迭代器
    for img in imgs:
        print(img.shape)  # (5, 250, 250, 3) (5, 250, 250, 3)
