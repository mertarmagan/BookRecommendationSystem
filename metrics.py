from random import shuffle

import pandas as pd

def read_df(path):
    return pd.read_csv(path, sep=",", low_memory=False)

def generate_sample_test():
    sample = [["2", "1", 5], ["5", "1", 3],
              ["2", "2", 3], ["3", "2", 9], ["7", "2", 6],
              ["4", "3", 5], ["6", "3", 7],
              ["2", "4", 2], ["5", "4", 0], ["8", "4", 1]]

    df = pd.DataFrame(sample, columns=["User-ID", "ISBN", "Book-Rating"])
    print("No. of rows: ", len(df))
    df.to_csv("./error/test1.csv", index=False, encoding="utf-8")

def generate_sample_pred():
    sample = [["2", "1", 2], ["5", "1", 1],
              ["2", "2", 8], ["3", "2", 9], ["7", "2", 6],
              ["4", "3", 5], ["6", "3", 7],
              ["2", "4", 2], ["5", "4", 0], ["8", "4", 1]]

    shuffle(sample)

    df = pd.DataFrame(sample, columns=["User-ID", "ISBN", "Book-Rating"])
    print("No. of rows: ", len(df))
    df.to_csv("./error/prediction1.csv", index=False, encoding="utf-8")

def mask(df, key, value):
    return df[df[key] == value]

# pred is prediction dataframe, test is test dataframe
def calculate_error(pred, test):
    # pred = read_df("./error/prediction1.csv")
    # test = read_df("./error/test1.csv")

    sum_error = 0

    for index, row in pred.iterrows():
        user = row["User-ID"]
        book = row["ISBN"]
        pred_rating = row["Book-Rating"]

        test_row = test[(test["User-ID"] == user) & (test["ISBN"] == book)].values.tolist()[0]

        test_rating = test_row[2]  # rating in test

        diff = pred_rating - test_rating
        sq_diff = diff ** 2

        sum_error += sq_diff

    error = sum_error ** (1/2)
    return error

