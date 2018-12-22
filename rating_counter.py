import pandas as pd
import labeler as lbl

def read_ratings(path):
    return pd.read_csv(path, sep=',', low_memory=False)

def find_mean(_dict):
    sum = 0
    for i in _dict:  # iterate over keys
        sum += _dict[i]

    mean = sum/len(_dict)

    return mean

def user_counter():
    ratings_tr = read_ratings("./modified-csv/train1_withID.csv")
    ratings_tst = read_ratings("./modified-csv/test1_withID.csv")

    ratings = pd.concat([ratings_tr, ratings_tst], ignore_index=True)
    users = ratings.iloc[:, 0].to_frame()

    users_list = sorted(users["User_ID"].tolist())
    _dict = {}

    for i in users_list:
        x = _dict.get(i, -1)
        if x == -1:
            _dict[i] = 1
        else:
            _dict[i] = x + 1

    return find_mean(_dict)

def user_rating_average(in_df):
    ratings = read_ratings("./modified-csv/sorted_user_"+in_df+".csv")
    users = ratings.iloc[:, 0:1]
    rats = ratings.iloc[:, 2:3]

    users_list = users["User_ID"].tolist()
    rat_list = rats["Rating"].tolist()

    main_dic = {}

    for i in range(len(users_list)):
        sub_dic = {}
        sub_dic["total"] = 0
        sub_dic["count"] = 0
        # sub_dic["avg"] = 0
        main_dic[users_list[i]] = sub_dic

    for i in range(len(users_list)):
        main_dic[users_list[i]]["count"] = main_dic[users_list[i]]["count"] + 1
        main_dic[users_list[i]]["total"] = main_dic[users_list[i]]["total"] + rat_list[i]

        main_dic[users_list[i]]["avg"] = main_dic[users_list[i]]["total"] / main_dic[users_list[i]]["count"]

    # print(len(main_dic.keys()))
    # print(main_dic[9])
    lbl.write_dict(main_dic, in_df + "-users-average")

def book_counter():
    ratings_tr = read_ratings("./modified-csv/train1_withID.csv")
    ratings_tst = read_ratings("./modified-csv/test1_withID.csv")

    ratings = pd.concat([ratings_tr, ratings_tst], ignore_index=True)
    books = ratings.iloc[:, 1].to_frame()

    books_list = sorted(books["Book_ID"].tolist())

    _dict = {}

    for i in books_list:
        x = _dict.get(i, -1)
        if x == -1:
            _dict[i] = 1
        else:
            _dict[i] = x + 1

    return find_mean(_dict)

def book_rating_average(in_df):
    ratings = read_ratings("./modified-csv/sorted_book_"+in_df+".csv")

    books = ratings.iloc[:, 1:2]
    rats = ratings.iloc[:, 2:3]

    books_list = books["Book_ID"].tolist()
    rat_list = rats["Rating"].tolist()

    main_dic = {}
    for i in range(len(books_list)):
        sub_dic = {}
        sub_dic["total"] = 0
        sub_dic["count"] = 0
        # sub_dic["avg"] = 0
        main_dic[books_list[i]] = sub_dic

    for i in range(len(books_list)):
        main_dic[books_list[i]]["count"] = main_dic[books_list[i]]["count"] + 1
        main_dic[books_list[i]]["total"] = main_dic[books_list[i]]["total"] + rat_list[i]

        main_dic[books_list[i]]["avg"] = main_dic[books_list[i]]["total"] / main_dic[books_list[i]]["count"]

    # print(len(main_dic.keys()))
    # print(main_dic[1])
    lbl.write_dict(main_dic,  in_df + "-books-average")

# Example Usage
# book_rating_average("train1")
# user_rating_average("train1")

# u = user_counter()
# print(u)
# b = book_counter()
# print(b)