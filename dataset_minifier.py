import pandas as pd
import labeler as lb
import numpy as np
from scipy.sparse import csr_matrix

def replaceISBN():
    df = pd.read_csv("./modified-csv/modified_ratings.csv")
    isbn_id = lb.read_dict("./json-outputs/isbn-to-id.json")

    def updateISBN(isbn):
        id = isbn_id[isbn]
        return id

    df.ISBN = df.ISBN.map(updateISBN)
    df.columns = ["User_ID", "Book_ID", "Rating"]

    df.to_csv("./modified-csv/ratings_with_id.csv", index=False, encoding="utf-8")

def sortDataset():
    df = pd.read_csv("./modified-csv/ratings_with_id.csv")
    # df["Book_ID"] = pd.to_numeric(df.Book_ID, errors="coerce")
    df.Book_ID = df.Book_ID.astype(int)
    df = df.sort_values(by='Book_ID', ascending=True, kind='quicksort')
    df.to_csv("./modified-csv/sorted_ratings.csv", index=False, encoding="utf-8")

def findIndexLength():
    df = pd.read_csv("./modified-csv/sorted_ratings.csv")

    id_list = df.Book_ID.tolist()

    main_dic = {}

    for i in range(len(df)):
        sub_dic = {}
        sub_dic["start"] = 0
        sub_dic["length"] = 1
        main_dic[id_list[i]] = sub_dic

    # print(main_dic)

    # count = range(0, len(df))

    for i in range(len(id_list)-1):
        if id_list[i] == id_list[i+1]:
            main_dic[id_list[i]]["length"] += 1
        else:
            main_dic[id_list[i+1]]["start"] = i+1
        # if id_list[i] == count[j]:
        #     main_dic[id_list[j]["start"]] = i
        # else:
        #     j += 1
        #     main_dic[id_list[j]["start"]] = i

    lb.write_dict(main_dic, "book-start-length")
    print(main_dic)

    # lb.write_dict()

# replaceISBN()
# sortDataset()
findIndexLength()