import numpy as np
import pandas as pd
import math
import similarity_finder as sim_find
import labeler as lb
import time
import matplotlib.pyplot as plt
import dataset_minifier as dm
import kfold
import rating_counter as rc

# from main.py import u   # Mean of every usr-book pair
# from main.py import x   # user index I'm lookin for
# from main.py import i   # book index I'm lookin for
# from .similarity_finder import similarity_finder

# sim_matrix = pd.read_cv("./ex.csv")

# usr_dev = pd.read_csv("./usr_dev.csv")  # user deviation
# bk_dev = pd.read_csv("./book_dev.csv")  # book deviation

def generate_dev(df, glob_mean, user_distinct, book_distinct, fold):
    books_avg = lb.read_dict("./json-outputs/train" + str(fold) + "-books-average.json")
    users_avg = lb.read_dict("./json-outputs/train" + str(fold) + "-users-average.json")

    book_dev = {}
    user_dev = {}

    for item in book_distinct:
        book_dev[item] = books_avg[str(item)]["avg"] - glob_mean
        # print(x_dict, sum/len(x_dict))

    for item in user_distinct:
        user_dev[item] = users_avg[str(item)]["avg"] - glob_mean

    # print(glob_mean)
    # print(user_dev)

    return user_dev, book_dev

def find_prediction(user_dev, book_dev, train, x, i, u, usl, index_len):
    
    devUser = user_dev.get(x, -1)
    if devUser == -1:
        devUser = 0

    devBook = book_dev.get(i, -1)
    if devBook == -1:
        devBook = 0

    bxi = u + devUser + devBook
    
    start = usl.get(x, -1)

    if start != -1:
        start = usl[str(x)]["start"]
        length = usl[str(x)]["length"]
    else:
        start = 0
        length = 0

    book_ratings = train.iloc[start:start+length, 1:].values

    # sims = np.array([[ None, None]])

    sum = 0
    total = 0

    for j in range(0, book_ratings.shape[0]):
        sim = sim_find.find_similarity(i, j, train, index_len)
        bxj = u + user_dev[x] + book_dev[book_ratings[j,0]]
        rxj = book_ratings[j, 1]
        if sim > 0:
            # sims = np.append(sims, np.array([[sim, sum]]), axis=0)
            sum = sum + (sim * (rxj - bxj))
            total = total + sim

    # sims = np.delete(sims, 0, 0)
    # k = int(sims.shape[0] / 2 + 1)
    # # print("k", k)
    # sims = sims[sims[:, 0].argsort()]
    # total = sims[sims.shape[0]-k:sims.shape[0],0]
    # ctr = sims[sims.shape[0]-k:sims.shape[0],1]

    # print("bxi: ", bxi)

    if total != 0:
        rxi = bxi + sum/total
    else:
        rxi = bxi

    return rxi

def RMSE(user_distinct, book_distinct, user_dev, book_dev, train, test, u):

    prediction = np.zeros(shape=(test.shape[0]), dtype="float")

    for i in range(test.shape[0]):
        prediction[i] = find_prediction(user_distinct, book_distinct, user_dev, book_dev, train, test[i, 0], test[i, 1], u)

    for i in range(prediction.shape[0]):
        mse = mse + (prediction[i] - test[i, 2]) ** 2
    
    rmse = math.sqrt(mse)
    print(rmse)
    return rmse

def conf_matix(user_dev, book_dev, train, test, u, usl, index_len, metric, fold):

    prediction = np.zeros(shape=(test.shape[0]), dtype="float")

    for i in range(test.shape[0]):
        prediction[i] = find_prediction(user_dev, book_dev, train, int(test.iloc[i].User_ID), int(test.iloc[i].Book_ID), u, usl, index_len)

    print("Prediction finished!")

    # th = 7

    precision = np.array([])
    recall = np.array([])
    accuracy = np.array([])

    for j in range(1,10):
        tp = 0
        tn = 0
        fp = 0
        fn = 0

        th = j
        for i in range(prediction.shape[0]):
            if prediction[i] >= th:
                if test.iloc[i].Rating >= th:
                    tp = tp + 1
                    # print("prediction:", prediction[i], "value:", test[i, 2], "user:", test[i, 0], "book:", test[i, 1])
                else:
                    # print("prediction:", prediction[i], "value:", test[i, 2], "user:", test[i,0], "book:", test[i,1])
                    fp = fp + 1
            else:
                if test.iloc[i].Rating < th:
                    tn = tn + 1
                else:
                    fn = fn + 1

        print("tp:", tp, "tn:", tn, "fp:", fp, "fn:", fn)

        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        accuracy = (tp + tn) / (fp + fn + tp + tn)

        metric[th-1][fold-1][0] = recall
        metric[th-1][fold-1][1] = precision
        metric[th-1][fold-1][2] = accuracy
    # print("accuracy:", (tp+ tn) / (fp + fn + tp + tn))
    # print("precision:", (tp / (tp + fp)))
    # print("recall:", (tp / (tp + fn)))

    # plt.plot([1,2,3,4,5,6,7,8,9,10], recall, label="recall")
    # plt.plot([1,2,3,4,5,6,7,8,9,10], precision, label="precision")
    # plt.plot([1,2,3,4,5,6,7,8,9,10], accuracy, label="accuracy")
    # plt.legend()
    # plt.show()

    return

def main():

    kfold.main()

    # train = pd.read_csv("./ex_similarity/sorted_train.csv", sep=",", low_memory=False)

    foldMetric = np.array([[ None, None, None]])

    train_arr = []
    test_arr = []

    usl_arr = []
    bsl_arr = []

    ud_arr = []
    bd_arr = []

    u_arr = []

    for fold in range(1, 11):

        dm.minify("train" + str(fold))
        dm.minify("test" + str(fold))

        rc.user_rating_average("train" + str(fold))
        rc.book_rating_average("train" + str(fold))

        train = pd.read_csv("./modified-csv/train" + str(fold) + "_withID.csv", sep=",", low_memory=False)
        test = pd.read_csv("./modified-csv/test" + str(fold) + "_withID.csv", sep=",", low_memory=False)

        train.columns = ["User_ID", "Book_ID", "Rating"]
        test.columns = ["User_ID", "Book_ID", "Rating"]

        train_arr.append(train)
        test_arr.append(test)

        bsl = lb.read_dict("./json-outputs/train" + str(fold) + "-book-start-length.json")
        usl = lb.read_dict("./json-outputs/train" + str(fold) + "-user-start-length.json")

        bsl_arr.append(bsl)
        usl_arr.append(usl)

        book_distinct = list(bsl_arr[fold-1].keys())
        user_distinct = list(usl_arr[fold-1].keys())

        book_distinct = list(map(int, book_distinct))
        user_distinct = list(map(int, user_distinct))

        u = np.average(train.iloc[:, 2])
        u_arr.append(u)

        book_dev, user_dev = generate_dev(train, u, user_distinct, book_distinct, fold)

        bd_arr.append(book_dev)
        ud_arr.append(user_dev)

    # for th in range(8,9):
    metric = np.zeros(shape=(10, 10, 3), dtype="float")
    for fold in range(1, 11):

        print("One fold finished: ", fold)

        # index_len = lb.read_dict("./json-outputs/book-start-length.json")
        conf_matix(ud_arr[fold-1], bd_arr[fold-1], train_arr[fold-1], test_arr[fold-1], u_arr[fold-1], usl_arr[fold-1], bsl_arr[fold-1], metric, fold)


    # avgAccuracy = np.average(foldMetric[1:,0:1])
    # avgPrecision = np.average(foldMetric[1:,1:2])
    # avgRecall = np.average(foldMetric[1:,2:3])

    # avgMetric = np.append(avgMetric, [avgAccuracy, avgPrecision, avgRecall])

    avg_rec = []
    avg_prec = []
    avg_acc = []
    for i in range(10):
        sum_rec = 0
        sum_prec = 0
        sum_acc = 0
        for j in range(10):
            sum_rec += metric[i][j][0]
            sum_prec += metric[i][j][1]
            sum_acc += metric[i][j][2]
        avg_rec.append(sum_rec/10)
        avg_prec.append(sum_prec/10)
        avg_acc.append(sum_acc/10)
    
    print("avg_rec: ", avg_rec)
    print("avg_prec: ", avg_prec)
    print("avg_acc: ", avg_acc)
    
main()