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



def generate_dev(df, u, user_distinct, book_distinct):

    x_dict = {}
    y_dict = {}

    x_arr = []
    y_arr = []

    book_dev = {}
    user_dev = {}

    x_yarr = {}

    ######## compress it all #########

    # print(user_distinct)
    # print("-------------")
    # print(book_distinct)

    for item in book_distinct:
        sum = 0
        x_dict = {}
        for index, row in df.iterrows():
            if row["ISBN"] == item:
                x_dict[row["User-ID"]] = row["Book-Rating"]
                temp = row["User-ID"]
                sum = sum + row["Book-Rating"]
        book_dev[item] = sum/len(x_dict) - u
        x_yarr[item] = sum/len(x_dict)
        # print(x_dict, sum/len(x_dict))

    # print(u)
    # print(book_dev)
    # print(x_yarr)

    for item in user_distinct:
        sum = 0
        y_dict = {}
        for index, row in df.iterrows():
            if row["User-ID"] == item:
                y_dict[row["ISBN"]] = row["Book-Rating"]
                sum = sum + row["Book-Rating"]
        user_dev[item] = sum/len(y_dict) - u
    
    # print(user_dev)

    # print("book-rating by user", y_arr)
    # print("+++++++++")
    # print("book-rating by user", x_dict)

    return user_dev, book_dev

def find_mean(_dict):
    sum = 0
    for i in _dict:  # iterate over keys
        sum += _dict[i]

    mean = sum/len(_dict)

    return mean

def find_prediction(user_distinct, book_distinct, user_dev, book_dev, train, x, i, u):
    bxi = u + user_dev[x] + book_dev[i]
    
    max1 = 0
    max2 = 0

    for item in book_distinct:
        sim = find_similarity(train, i, item)
        print(item, sim)

    # for j in range(0, book_distinct[0]):
    #     if i is not j:
    #         if max2 == max1:
    #             max2 = find_similarity(train, i, j)
    #         else:
    #             max1 = find_similarity(train, i, j)
    #             if max1 < max2:
    #                 temp = max2
    #                 max2 = max1
    #                 max1 = temp

    # print(max1, max2)

    # for j in range(0, sim_matrix.shape[0]):
    #     if not np.isnan(sim_matrix.iloc[i,j]):
    #         bxj = u + usr_dev.iloc[x] + bk_dev.iloc[j]
    #         # sum = sum + (sim_matrix.iloc[i,j] * (ratings_matrix.iloc[x,j] - bxj)

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

    # print(mean_x, mean_y)

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

    book_distinct = np.array([], dtype="int")
    user_distinct = np.array([], dtype="int")

    user_distinct = np.append(user_distinct, [train.iloc[0, 0]])
    book_distinct = np.append(book_distinct, [train.iloc[0, 1]])

    for i in range(0, train.shape[0]):
        if train.iloc[i, 0] not in user_distinct:
            user_distinct = np.append(user_distinct, [train.iloc[i, 0]])
        if train.iloc[i, 1] not in book_distinct:
            book_distinct = np.append(book_distinct, [train.iloc[i, 1]])

    u = np.average(train.iloc[:, 2])

    # print(generate_dev(train, u))

    dev = generate_dev(train, u, user_distinct, book_distinct)
    book_dev = dev[1]
    user_dev = dev[0]

    find_prediction(user_distinct, book_distinct, user_dev, book_dev, train, 5, 1, u)

    # print(u)

    # print(find_similarity(train, 1, 2))
    # find_prediction()
    # similarity_finder()
    # generate_devs()
    # find_prediction()

main()