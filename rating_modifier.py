import pandas as pd
import labeler as lbl
import csv

def read_ratings():
    return pd.read_csv("./modified_ratings.csv", sep=';', low_memory=False)

def main():
    ratings = read_ratings()

    ratings = ratings.sample(frac=1, replace=True)

    # ratings.to_csv("./shuffled_ratings.csv", index=False, encoding="utf-8", quoting=csv.QUOTE_NONE, quotechar="", escapechar="\\")

    isbn_dict = lbl.read_dict("./isbn-to-id.json")
    book_list = list(isbn_dict.keys())

    _list = []

    for index, row in ratings.iterrows():
        if row["ISBN"] in book_list:
            _list.append(row)

    new_df = pd.DataFrame(_list, columns=['User-ID', 'ISBN', 'Book-Rating'])

    # print(new_df.head())
    # print(len(new_df))
    new_df.to_csv("./modified_ratings.csv", index=False, encoding="utf-8")
    return
# main()