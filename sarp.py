import numpy as np
import pandas as pd
# from main.py import u   # Mean of every usr-book pair
# from main.py import x   # user index I'm lookin for
# from main.py import i   # book index I'm lookin for
# from .similarity_finder import similarity_finder

# import similarity_finder

# sim_matrix = pd.read_cv("./ex.csv")

# usr_dev = pd.read_csv("./usr_dev.csv")  # user deviation
# bk_dev = pd.read_csv("./book_dev.csv")  # book deviation


def generate_dev(df):

    book_distinct = np.array([], dtype="int")
    user_distinct = np.array([], dtype="int")

    x_dict = {}
    y_dict = {}

    x_arr = []
    y_arr = []

    user_distinct = np.append(user_distinct, [df.iloc[0, 0]])
    book_distinct = np.append(book_distinct, [df.iloc[0, 1]])

    for i in range(0, df.shape[0]):
        if df.iloc[i, 0] not in user_distinct:
            user_distinct = np.append(user_distinct, [df.iloc[i, 0]])
        if df.iloc[i, 1] not in book_distinct:
            book_distinct = np.append(book_distinct, [df.iloc[i, 1]])

    # print(user_distinct)
    # print("-------------")
    # print(book_distinct)

    for item in book_distinct:
        for index, row in df.iterrows():
            if row["ISBN"] == item:
                x_dict[row["User-ID"]] = row["Book-Rating"]


    for item in user_distinct:
        for index, row in df.iterrows():
            if row["User-ID"] == item:
                y_dict[row["ISBN"]] = row["Book-Rating"]
        print(y_dict)



    # print("book-rating by user", y_arr)
    # print("+++++++++")
    # print("book-rating by user", x_dict)

def find_mean(_dict):
    sum = 0
    for i in _dict:  # iterate over keys
        sum += _dict[i]

    mean = sum/len(_dict)

    return mean

def find_prediction(usr_dev, bk_dev, x, i):
    bxi = u + usr_dev.iloc[x] + bk_dev.iloc[i]
    sum = 0
    for j in range(0, sim_matrix.shape[0]):
        if not np.isnan(sim_matrix.iloc[i,j]):
            bxj = u + usr_dev.iloc[x] + bk_dev.iloc[j]
            # sum = sum + (sim_matrix.iloc[i,j] * (ratings_matrix.iloc[x,j] - bxj)

def find_similarity(df, x, y):
    x_dict = {}
    y_dict = {}

    for index, row in df.iterrows():
        if row["ISBN"] == x:
            x_dict[row["User-ID"]] = row["Book-Rating"]

    for index, row in df.iterrows():
        if row["ISBN"] == y:
            y_dict[row["User-ID"]] = row["Book-Rating"]

    print(x_dict)
    # print(y_dict)

    res_x_dict = {}
    res_y_dict = {}
    # Finding the intersecting users
    for item in x_dict.keys():
        if item in y_dict:
            res_y_dict[item] = y_dict[item]
            res_x_dict[item] = x_dict[item]

    # print(res_x_dict)
    # print(res_y_dict)

    mean_x = find_mean(x_dict)
    mean_y = find_mean(y_dict)

    print(mean_x, mean_y)

    for i in x_dict:
        val = x_dict[i]
        x_dict[i] = val - mean_x

    for i in y_dict:
        val = y_dict[i]
        y_dict[i] = val - mean_y

    # print(x_dict)
    # print(y_dict)

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

    # if sum_x == 0:
    #     sum_x = 1
    #
    # if sum_y == 0:
    #     sum_y = 1

    sim = sum / (sum_x * sum_y)
    return sim


def main():
    train = pd.read_csv("./ex_similarity/train.csv", sep=",", low_memory=False)

    u = np.average(train.iloc[:, 2])

    generate_dev(train)

    # print(u)

    # print(find_similarity(train, 1, 2))
    # find_prediction()
    # similarity_finder()
    # generate_devs()
    # find_prediction()

main()