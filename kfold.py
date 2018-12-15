import pandas as pd
import rating_modifier as rm


def kfold_split(dataset, folds = 10):
    df = pd.DataFrame(dataset, columns=['User-ID', 'ISBN', 'Book-Rating'])
    fold_size = len(df) / folds
    for i in range(folds):
        test = df.iloc[int(i*fold_size):int(i*fold_size+fold_size), :]
        train = df.iloc[int(i*fold_size+fold_size):len(dataset), :]
        if i is not 0:
            train = pd.concat((train, df.iloc[0:int(i*fold_size), :]))

        test.to_csv("./partition/test{}.csv".format(i+1), index=False, encoding="utf-8")
        train.to_csv("./partition/train{}.csv".format(i+1), index=False, encoding="utf-8")

def main():
    df = pd.read_csv("./modified-csv/shuffled_ratings.csv", sep=',', low_memory=False)
    kfold_split(df, 10)

# main()
