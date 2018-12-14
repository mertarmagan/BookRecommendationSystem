import pandas as pd
import labeler as lb

def replaceISBN():
    df = pd.read_csv("./modified-csv/modified_ratings.csv")
    isbn_id = lb.read_dict("./json-outputs/isbn-to-id.json")

    def updateISBN(isbn):
        id = isbn_id[isbn]
        return id

    df.ISBN = df.ISBN.map(updateISBN)
    df.columns = ["User_ID", "Book_ID", "Rating"]

    df.to_csv("./modified-csv/ratings_with_id.csv", index=False, encoding="utf-8")

def sortDatasetBook():
    df = pd.read_csv("./modified-csv/ratings_with_id.csv")
    df.Book_ID = df.Book_ID.astype(int)
    df = df.sort_values(by='Book_ID', ascending=True, kind='quicksort')
    df.to_csv("./modified-csv/sorted_book_ratings.csv", index=False, encoding="utf-8")

def sortDatasetExample():
    df = pd.read_csv("./ex_similarity/train.csv")
    df.Book_ID = df.Book_ID.astype(int)
    df = df.sort_values(by='Book_ID', ascending=True, kind='quicksort')
    df.to_csv("./ex_similarity/sorted_train.csv", index=False, encoding="utf-8")

def sortDatasetUser():
    df = pd.read_csv("./modified-csv/ratings_with_id.csv")
    df.User_ID= df.User_ID.astype(int)
    df = df.sort_values(by='User_ID', ascending=True, kind='quicksort')
    df.to_csv("./modified-csv/sorted_user_ratings.csv", index=False, encoding="utf-8")

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

def findIndexLengthBooks():
    df = pd.read_csv("./modified-csv/sorted_book_ratings.csv")

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

    lb.write_dict(main_dic, "book-start-length")
    print(main_dic)

def findIndexLengthUsers():
    df = pd.read_csv("./modified-csv/sorted_user_ratings.csv")

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

    lb.write_dict(main_dic, "user-start-length")
    print(main_dic)

# replaceISBN()
# sortDatasetBook()
# findIndexLengthBooks()
# sortDatasetUser()
# findIndexLengthUsers()
# sortDatasetExample()
# findIndexLengthExample()