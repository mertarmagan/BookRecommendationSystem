import pandas as pd
import labeler as lbl

def read_ratings():
    return pd.read_csv("./modified_ratings.csv", sep=',', low_memory=False)

def user_counter():
    ratings = read_ratings()
    users = ratings.iloc[:, 0:1]

    users_list = sorted(users["User-ID"].tolist())
    _dict = {}

    for i in users_list:
        x = _dict.get(i, -1)
        if x == -1:
            _dict[i] = 1
        else:
            _dict[i] = x + 1

    print(len(_dict.keys()))
    # print(_dict)
    # lbl.write_dict(_dict, "user-ratings-count")

def book_counter():
    ratings = read_ratings()
    books = ratings.iloc[:, 1:2]
    books_list = sorted(books["ISBN"].tolist())

    _dict = {}

    for i in books_list:
        x = _dict.get(i, -1)
        if x == -1:
            _dict[i] = 1
        else:
            _dict[i] = x + 1

    print(len(_dict.keys()))
    # lbl.write_dict(_dict, "book-ratings-count")

book_counter()