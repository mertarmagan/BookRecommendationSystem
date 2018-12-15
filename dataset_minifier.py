import pandas as pd
import labeler as lb

def replaceISBN(in_df):
    df = pd.read_csv("./modified-csv/" + in_df + ".csv")
    isbn_id = lb.read_dict("./json-outputs/isbn-to-id.json")

    def updateISBN(isbn):
        id = isbn_id[isbn]
        return id

    df.ISBN = df.ISBN.map(updateISBN)
    df.columns = ["User_ID", "Book_ID", "Rating"]

    df.to_csv("./modified-csv/" + in_df + "_withID.csv", index=False, encoding="utf-8")

def sortDatasetBook(in_df):
    df = pd.read_csv("./modified-csv/" + in_df + "_withID.csv")
    df.Book_ID = df.Book_ID.astype(int)
    df = df.sort_values(by='Book_ID', ascending=True, kind='quicksort')
    df.to_csv("./modified-csv/sorted_book_" + in_df + ".csv", index=False, encoding="utf-8")

def sortDatasetExample():
    df = pd.read_csv("./ex_similarity/train.csv")
    df.Book_ID = df.Book_ID.astype(int)
    df = df.sort_values(by='Book_ID', ascending=True, kind='quicksort')
    df.to_csv("./ex_similarity/sorted_train.csv", index=False, encoding="utf-8")

def sortDatasetUser(in_df):
    df = pd.read_csv("./modified-csv/" + in_df + "_withID.csv")
    df.User_ID= df.User_ID.astype(int)
    df = df.sort_values(by='User_ID', ascending=True, kind='quicksort')
    df.to_csv("./modified-csv/sorted_user_" + in_df + ".csv", index=False, encoding="utf-8")

def findIndexLengthExample():
    df = pd.read_csv("./ex_similarity/sorted_train.csv")

    id_list = df.Book_ID.tolist()

    main_dic = {}

    for i in range(len(df)):
        sub_dic = {}
        sub_dic["start"] = 0
        sub_dic["length"] = 1
        main_dic[id_list[i]] = sub_dic

    for i in range(len(id_list) - 1):
        if id_list[i] == id_list[i + 1]:
            main_dic[id_list[i]]["length"] += 1
        else:
            main_dic[id_list[i + 1]]["start"] = i + 1

    lb.write_dict(main_dic, "train-start-length")
    print(main_dic)

def findIndexLengthBooks(in_df):
    df = pd.read_csv("./modified-csv/sorted_book_" + in_df + ".csv")

    id_list = df.Book_ID.tolist()

    main_dic = {}

    for i in range(len(df)):
        sub_dic = {}
        sub_dic["start"] = 0
        sub_dic["length"] = 1
        main_dic[id_list[i]] = sub_dic

    for i in range(len(id_list)-1):
        if id_list[i] == id_list[i+1]:
            main_dic[id_list[i]]["length"] += 1
        else:
            main_dic[id_list[i+1]]["start"] = i+1

    lb.write_dict(main_dic, in_df + "-book-start-length")

def findIndexLengthUsers(in_df):
    df = pd.read_csv("./modified-csv/sorted_user_" + in_df + ".csv")

    id_list = df.User_ID.tolist()

    main_dic = {}

    for i in range(len(df)):
        sub_dic = {}
        sub_dic["start"] = 0
        sub_dic["length"] = 1
        main_dic[id_list[i]] = sub_dic

    for i in range(len(id_list)-1):
        if id_list[i] == id_list[i+1]:
            main_dic[id_list[i]]["length"] += 1
        else:
            main_dic[id_list[i+1]]["start"] = i+1

    lb.write_dict(main_dic, in_df + "-user-start-length")

def minify(in_df):
    replaceISBN(in_df)
    sortDatasetBook(in_df)
    findIndexLengthBooks(in_df)
    sortDatasetUser(in_df)
    findIndexLengthUsers(in_df)

# minify("train1")
# replaceISBN("train1")
# sortDatasetBook("train1")
# findIndexLengthBooks()
# sortDatasetUser()
# findIndexLengthUsers()
# sortDatasetExample()
# findIndexLengthExample()