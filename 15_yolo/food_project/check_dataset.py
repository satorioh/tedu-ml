# UNIMIB2016
# ├── UNIMIB2016-annotations
# │   ├── check_dataset.py <--
# │   └── formatted_annotations
# └── images

# check_dataset.py

import os

# path of formatted_annotations
f_path = os.path.join(os.getcwd(), '../source/annotations/formatted_annotations')

# path of images
img_path = os.path.join(os.getcwd(), '../source/original')


def check_dataset():
    annotations = [i[:-4] for i in os.listdir(f_path)]
    imgs = [i[:-4] for i in os.listdir(img_path)]

    for annotation in annotations:
        label = annotation + '.txt'
        label_path = os.path.join(f_path, label)

        try:
            if annotation not in imgs:
                # remove annotation which is not in images
                print('not found image: {}, remove its annotation'.format(annotation))
                print(label_path)
                # 将文件名带(0)的图片重命名为annotation + '.jpg'
                old_img_path = os.path.join(img_path, annotation + '(0).jpg')
                new_img_path = os.path.join(img_path, annotation + '.jpg')
                os.rename(old_img_path, new_img_path)
                raise FileExistsError

            else:
                # check extra spaces in a line
                with open(label_path) as f:
                    lines = f.readlines()
                    for line in lines:
                        item = line.split()
                        if len(item) > 9:
                            print('wrong label format: {}, {}'.format(annotation, line))
                            raise FileExistsError

        except FileExistsError:
            pass
    # os.remove(label_path)
    # print('os.remove({})'.format(label_path))


if __name__ == '__main__':
    check_dataset()
