import csv
import numpy as np
import matplotlib.pyplot as plt


def import_file():
    with open('full-gt.csv', 'r') as f:
        reader = csv.reader(f)
        file = np.array(list(reader))
        file = np.delete(file, 0, axis=0)
        return file[:, 0:7]


def find_min_max(data):
    data = data[:, 1:5]
    data = np.array(data, dtype=float)
    indexes = []
    min_x, min_y, max_x, max_y = 9999, 9999, 0, 0
    for idx, row in enumerate(data):
        x1 = row[0]
        y1 = row[1]
        x2 = row[2] + x1
        y2 = row[3] + y1

        if x1 == 0 or y1 == 0:
            indexes.append(idx)
            continue

        if x1 < min_x:
            min_x = x1
        if y1 < min_y:
            min_y = y1
        if x2 > max_x:
            max_x = x2
        if y1 > max_y:
            max_y = y2

    return min_x, min_y, max_x, max_y


def unique(my_list):
    b = []
    for i in my_list:
        # if its not in b, it's unique
        if i not in b:
            b.append(i)
    return b


def encode_classes(y):
    classes = unique(y)
    # will be an index in accordance to its class row in classes
    return [classes.index(row) for row in y]


def add_column(X, y):
    return_list = list()
    # keep index do we can add to correct row
    for idx, val in enumerate(X):
        return_list.append(val)
        return_list[idx] = np.append(return_list[idx], y[idx])
    return return_list


def analyse_classes(file):
    classes, counts = np.unique(file[:, 5], return_counts=True)
    class_summary = dict(zip(classes, counts))
    class_sorted = sorted(class_summary.items(), key=lambda x: x[0], reverse=False)
    x, y = zip(*[(i[0], i[1]) for i in class_sorted])
    print(x)
    return
    class_sorted = filter(lambda i: i[1] > 1000, class_sorted)
    x, y = zip(*[(i[0], i[1]) for i in class_sorted])

    y_pos = np.arange(len(x))

    plt.bar(y_pos, y, align='center', alpha=0.5)
    plt.xticks(y_pos, x)
    plt.xticks(rotation=90)
    plt.ylabel('Sign Type')
    plt.title('Distribution of Sign classes')

    plt.show()
    # print(x)

if __name__ == '__main__':
    # print(find_min_max(import_file()))
    analyse_classes(import_file())

