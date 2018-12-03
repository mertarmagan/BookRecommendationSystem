import pandas as pd

def read_train(path):
    return pd.read_csv(path, sep=",", low_memory=False)

def find_mean(_dict):
    sum = 0
    for i in _dict:  # iterate over keys
        sum += _dict[i]

    mean = sum/len(_dict)

    return mean

def sample_generator():  # Example from slide 43
    sample = [["1", "1", 1], ["1", "3", 2], ["1", "6", 1],
              ["2", "3", 4], ["2", "4", 2],
              ["3", "1", 3], ["3", "2", 5], ["3", "4", 4], ["3", "5", 4], ["3", "6", 3],
              ["4", "2", 4], ["4", "3", 1], ["4", "5", 3],
              ["5", "3", 2], ["5", "4", 5], ["5", "5", 4], ["5", "6", 3],
              ["6", "1", 5], ["6", "5", 2],
              ["7", "2", 4], ["7", "3", 3],
              ["8", "4", 4], ["8", "6", 2],
              ["9", "1", 5], ["9", "3", 4],
              ["10", "2", 2], ["10", "3", 3],
              ["11", "1", 4], ["11", "2", 1], ["11", "3", 5], ["11", "4", 2], ["11", "5", 2], ["11", "6", 4],
              ["12", "2", 3], ["12", "5", 5]]

    df = pd.DataFrame(sample, columns=["User-ID", "ISBN", "Book-Rating"])
    print("No. of rows: ", len(df))
    df.to_csv("./ex_similarity/train.csv", index=False, encoding="utf-8")

def sample_generator2():  # Example from slide 37
    sample = [["1", "1", 4], ["1", "4", 5], ["1", "5", 1],
              ["2", "1", 5], ["2", "2", 2], ["2", "3", 5],
              ["3", "4", 2], ["3", "5", 4], ["3", "6", 5],
              ["4", "2", 3], ["4", "7", 3]]

    df = pd.DataFrame(sample, columns=["User-ID", "ISBN", "Book-Rating"])
    print("No. of rows: ", len(df))
    df.to_csv("./ex_similarity/train2.csv", index=False, encoding="utf-8")

def find_similarity(df, x, y):
    x_dict = {}
    y_dict = {}

    for index, row in df.iterrows():
        if row["ISBN"] == x:
            x_dict[row["User-ID"]] = row["Book-Rating"]

    for index, row in df.iterrows():
        if row["ISBN"] == y:
            y_dict[row["User-ID"]] = row["Book-Rating"]

    # print(x_dict)

    res_x_dict = {}
    res_y_dict = {}
    # Finding the intersecting users
    for item in x_dict.keys():
        if item in y_dict:
            res_y_dict[item] = y_dict[item]
            res_x_dict[item] = x_dict[item]

    print(res_x_dict)
    print(res_y_dict)

    mean_x = find_mean(x_dict)
    mean_y = find_mean(y_dict)

    for i in x_dict:
        val = x_dict[i]
        x_dict[i] = val - mean_x

    for i in y_dict:
        val = y_dict[i]
        y_dict[i] = val - mean_y

    print(x_dict)
    print(y_dict)

    sum = 0
    sum_x = 0
    sum_y = 0

    for i in x_dict:
        sum_x += x_dict[i] ** 2

    for i in y_dict:
        sum_y += y_dict[i] ** 2

    for i in res_x_dict:
        sum += x_dict[i] * y_dict[i]

    sum_x = sum_x ** (1/2)
    sum_y = sum_y ** (1/2)

    sim = sum / (sum_x * sum_y)

    # print("sum:", sum)
    return sim

def main():
    train = read_train("./ex_similarity/train.csv")
    print(find_similarity(train, 1, 3))

main()