import pandas as pd
import labeler as lbl

def read_ratings():
    return pd.read_csv("./modified_ratings.csv", sep=',', low_memory=False)

def main():
    ratings = read_ratings()
    users = ratings.iloc[:, 0:1]

    users_list = sorted(users["User-ID"].tolist())
    _dict = {}

    for i in users_list:
        x = _dict.get(i, -1)
        if x == -1:
            _dict[i] = 1
        else:
            _dict[i] = x + 1

    # print(_dict)
    lbl.write_dict(_dict, "user-ratings-count")

main()