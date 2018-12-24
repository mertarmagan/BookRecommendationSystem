import pandas as pd
import numpy as np
import dataset_minifier as dm
import rating_counter as rc
import similarity_finder as sf
import labeler as lb

def trainAppend(new_df):
    isbn_list = new_df.iloc[:, 1].tolist()
    # print(isbn_list)

    old = pd.read_csv("./modified-csv/train1.csv", sep=",", low_memory=False)
    final_df = pd.concat([old, new_df])
    # print(final_df.shape)
    # print(final_df.tail())

    final_df.to_csv("./modified-csv/train1_demo.csv", index=False, encoding="utf-8")

    return isbn_list

def modifyTest(isbn_list):
    test = pd.read_csv("./modified-csv/test1.csv", sep=",", low_memory=False)
    # test.columns = ["User_ID", "Book_ID", "Rating"]
    test.loc[test['User-ID'] < 999999, 'User-ID'] = 999999

    drop_list = []
    for index, row in test.iterrows():
        if row["ISBN"] in isbn_list:
            drop_list.append(index)
    
    # print(test.shape[0])

    test = test.drop(drop_list, axis=0)

    # print(test.shape[0])

    test.to_csv("./modified-csv/test1_modified.csv", index=False, encoding="utf-8")
    return

def generate_dev(glob_mean, user_distinct, book_distinct):
    books_avg = lb.read_dict("./json-outputs/train1_demo-books-average.json")
    users_avg = lb.read_dict("./json-outputs/train1_demo-users-average.json")

    book_dev = {}
    user_dev = {}

    for item in book_distinct:
        book_dev[item] = books_avg[str(item)]["avg"] - glob_mean
        # print(x_dict, sum/len(x_dict))

    for item in user_distinct:
        user_dev[item] = users_avg[str(item)]["avg"] - glob_mean

    return user_dev, book_dev

def find_prediction(user_dev, book_dev, train_user, train_book, x, i, u, usl, index_len):
    devUser = user_dev.get(x, -1)
    if devUser == -1:
        devUser = 0

    devBook = book_dev.get(i, -1)
    if devBook == -1:
        devBook = 0

    bxi = u + devUser + devBook
    
    info = usl.get(str(x), -1)

    if info != -1:
        start = info["start"]
        length = info["length"]
    else:
        start = 0
        length = 0

    book_ratings = train_user.iloc[start:start+length].values

    sum = 0
    total = 0
    for j in range(0, book_ratings.shape[0]):
        bookY = book_ratings[j, 1]
        devBookY = user_dev.get(bookY, -1)
        if devBookY == -1:
            devBookY = 0

        sim = sf.find_similarity(i, bookY, train_book, index_len, devBook+u, devBookY+u)
        bxj = u + user_dev[x] + book_dev[bookY]
        rxj = book_ratings[j, 2]
        if sim > 0:
            # print(sim)
            sum = sum + (sim * (rxj - bxj))
            total = total + sim

    if total != 0:
        rxi = bxi + sum/total
    else:
        rxi = bxi

    return rxi

def conf_matix(user_dev, book_dev, train_user, train_book, test, u, usl, bsl):

    id_arr = np.zeros(shape=(5), dtype="int")
    count = 0

    for i in range(test.shape[0]):
        pred = find_prediction(user_dev, book_dev, train_user, train_book, int(test.iloc[i].User_ID), int(test.iloc[i].Book_ID), u, usl, bsl)
        if pred > 3:
            # print(pred, int(test.iloc[i].Book_ID))
            id_arr[count] = int(test.iloc[i].Book_ID)
            count += 1
        if count == 5:
            break

    converter(id_arr)

    print("Prediction finished!")
    return

def converter(id_arr):
    books = pd.read_csv("./modified-csv/books.csv", sep=";", low_memory=False)
    id_list = lb.read_dict("./json-outputs/id-to-isbn.json")
    isbn_list = []

    for i in id_arr:
        isbn_list.append(id_list[str(i)])
    
    count = 0
    for index, row in books.iterrows():
        if row["ISBN"] in isbn_list:
            print("Book Name:", row["Book-Title"], ", Author:", row["Book-Author"])
            count += 1
        if count == 5:
            break
    # print(isbn_list)
    return

def main():

    new_df = pd.read_csv("./modified-csv/new.csv", sep=",", low_memory=False, dtype="str")

    isbn_list = trainAppend(new_df)
    modifyTest(isbn_list)

    dm.minify("train1_demo")
    dm.minify("test1_modified")
    
    rc.user_rating_average("train1_demo")
    rc.book_rating_average("train1_demo")

    train_user = pd.read_csv("./modified-csv/sorted_user_train1_demo.csv", sep=",", low_memory=False)
    train_book = pd.read_csv("./modified-csv/sorted_book_train1_demo.csv", sep=",", low_memory=False)
    test = pd.read_csv("./modified-csv/sorted_user_test1_modified.csv", sep=",", low_memory=False)

    train_user.columns = ["User_ID", "Book_ID", "Rating"]
    train_book.columns = ["User_ID", "Book_ID", "Rating"]
    test.columns = ["User_ID", "Book_ID", "Rating"]

    bsl = lb.read_dict("./json-outputs/train1_demo-book-start-length.json")
    usl = lb.read_dict("./json-outputs/train1_demo-user-start-length.json")

    book_distinct = list(bsl.keys())
    user_distinct = list(usl.keys())

    book_distinct = list(map(int, book_distinct))
    user_distinct = list(map(int, user_distinct))

    u = np.average(train_user.iloc[:, 2])

    user_dev, book_dev = generate_dev(u, user_distinct, book_distinct)

    conf_matix(user_dev, book_dev, train_user, train_book, test, u, usl, bsl)

    return

main()