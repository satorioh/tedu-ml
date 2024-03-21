# UNIMIB2016
# ├── UNIMIB2016-annotations
# │   ├── check_dataset.py
# │   ├── class_count.py <--
# │   └── formatted_annotations
# └── images

# class_count.py

import os
import pandas as pd

# formatted_annotations path
path = os.path.join(os.getcwd(), '../source/annotations/formatted_annotations')

# output path
output = os.path.join(os.getcwd(), './class_counts_result.csv')

# read file list of formatted_annotations
annotations = os.listdir(path)

if __name__ == '__main__':
    labels = []
    for annotation in annotations:
        with open(os.path.join(path, annotation)) as file:
            for line in file:
                item = line.split()
                cls = item[0]
                labels.append(cls)
    counts = pd.Series(labels).value_counts()
    counts.to_csv(output, header=False)
