# UNIMIB2016
# ├── UNIMIB2016-annotations
# │   ├── check_dataset.py
# │   ├── class_count.py
# │   ├── toYolo.py <--
# │   ├── class_counts_result.csv
# │   └── formatted_annotations (1005)
# ├── labels
# └── images (1005)

# toYolo.py

import os
from PIL import Image

# formatted_annotations path
path = os.path.join(os.getcwd(), '../source/annotations/formatted_annotations')

# path of images
img_path = os.path.join(os.getcwd(), '../source/original')

# output path
output_path = os.path.join(os.getcwd(), './labels')

# class count file path
class_file_path = os.path.join(os.getcwd(), './class_counts_result.csv')


def convert_box(size, box):
    # convert VOC to yolo format
    # box: [xmin, xmax, ymin, ymax]
    dw, dh = 1. / size[0], 1. / size[1]  # 归一化比例
    x, y, w, h = (box[0] + box[1]) / 2.0, (box[2] + box[3]) / 2.0, box[1] - box[0], box[3] - box[2]  # 中心点坐标和宽高
    return [x * dw, y * dh, w * dw, h * dh]  # 归一化后的坐标


def convert_bbox(ibb):
    # convert ibb to VOC format
    # ibb = [x1,y1,x2,y2,x3,y3,x4,y4]
    X = ibb[0::2]  # [x1,x2,x3,x4]
    Y = ibb[1::2]  # [y1,y2,y3,y4]
    xmin = min(X)
    ymin = min(Y)
    xmax = max(X)
    ymax = max(Y)
    return xmin, ymin, xmax, ymax


def get_classes():
    # output: class list
    cf = open(class_file_path, 'r')
    clss = [line.split(',')[0] for line in cf.readlines()]
    cf.close()
    return clss


def toYolo():
    # read file list of formatted_annotations
    annotations = os.listdir(path)

    # get class list
    clss = get_classes()

    # convert every annotation in ./formatted_annotations/ to yolo format
    for annotation in annotations:

        with open(os.path.join(path, annotation)) as file, open(os.path.join(output_path, annotation), 'w') as opfile:

            # read img
            img_f_path = os.path.join(img_path, annotation[:-4] + '.jpg')
            img = Image.open(img_f_path)

            # get img size
            size = img.size

            # process every item in ./formatted_annotations/*.txt
            for line in file:
                item = line.split(' ')

                # get class num
                cls = item[0]
                cls_num = clss.index(cls)

                # get bbox coordinates
                item_bounding_box = list(map(float, item[1:]))
                xmin, ymin, xmax, ymax = convert_bbox(item_bounding_box)
                b = [xmin, xmax, ymin, ymax]
                bb = convert_box(size, b)

                # append item to output file: ../labels/*.txt
                item_str = list(map(str, [cls_num] + bb))
                line_yolo = ' '.join(item_str)
                opfile.write(line_yolo + '\n')

            print(annotation)


if __name__ == '__main__':
    toYolo()
