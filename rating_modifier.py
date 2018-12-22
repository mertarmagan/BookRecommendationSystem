import pandas as pd
import labeler as lbl
import numpy as np
import csv
from sklearn.utils import shuffle

def read_ratings(path):
    return pd.read_csv(path, sep=';', low_memory=False)

def main():
    ratings = read_ratings("./modified-csv/modified_ratings.csv")
    # ratings.to_csv("./modified-csv/modified_ratings.csv", index=False, encoding="utf-8")
    # ratings.reindex(np.random.permutation(ratings.index))

    # ratings = shuffle(ratings)

    # ratings.to_csv("./shuffled_ratings.csv", index=False, encoding="utf-8", quoting=csv.QUOTE_NONE, quotechar="", escapechar="\\")

    # shuffle = read_ratings("./shuffled_ratings.csv")
    # print("Ratings 1", shuffle.iloc[983442])
    # print("Ratings 1", shuffle.iloc[524738])

    isbn_dict = lbl.read_dict("./json-outputs/isbn-to-id.json")
    book_list = list(isbn_dict.keys())

    _list = []

    for index, row in ratings.iterrows():
        if row["ISBN"] in book_list:
            _list.append(row)

    new_df = pd.DataFrame(_list, columns=['User-ID', 'ISBN', 'Book-Rating'])

    # print(len(new_df))
    new_df.to_csv("./modified-csv/modified_ratings.csv", index=True, encoding="utf-8")
    return

# main()