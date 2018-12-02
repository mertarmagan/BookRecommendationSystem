import pandas as pd
import numpy as np
import json as js

def read_books():
    return pd.read_csv("./books.csv", sep=';', low_memory=False)

def write_dict(dict1):
    json = js.dumps(dict1)
    f = open("id-to-isbn.json", "w")
    f.write(json)
    f.close()

def read_dict(path):
    f = open(path)
    dict1 = js.load(f)
    f.close()
    return dict1

def main():
    books = read_books()
    isbn = books.iloc[:, 0:1]
    dict1 = {}
    isbn_list = isbn['ISBN'].tolist()

    index = 0
    for i in isbn_list:
        dict1[index] = i
        index += 1

    # a = read_dict("./isbn-to-id.json")
    # n_items = {k: a[k] for k in list(a)[:10]}

    # print(dict['0767409752'])
    # print(isbn_list)
    # print(isbn.head())

main()

