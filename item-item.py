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

def generate_dev(glob_mean, user_distinct, book_distinct, fold):
    books_avg = lb.read_dict("./json-outputs/train" + str(fold) + "-books-average.json")
    users_avg = lb.read_dict("./json-outputs/train" + str(fold) + "-users-average.json")

    book_dev = {}
    user_dev = {}

    for item in book_distinct:
        book_dev[item] = books_avg[str(item)]["avg"] - glob_mean
        # print(x_dict, sum/len(x_dict))

    for item in user_distinct:
        user_dev[item] = users_avg[str(item)]["avg"] - glob_mean

    return user_dev, book_dev

def find_prediction(user_dev, book_dev, train_user, train_book, x, i, u, usl, index_len):
    
    devUser = user_dev.get(x, -1)
    if devUser == -1:
        devUser = 0

    devBook = book_dev.get(i, -1)
    if devBook == -1:
        devBook = 0

    bxi = u + devUser + devBook
    
    info = usl.get(x, -1)

    if info != -1:
        start = info["start"]
        length = info["length"]
    else:
        start = 0
        length = 0

    book_ratings = train_user.iloc[start:start+length].values

    sum = 0
    total = 0
    for j in range(0, book_ratings.shape[0]):
        bookY = book_ratings[j, 1]
        devBookY = user_dev.get(bookY, -1)
        if devBookY == -1:
            devBookY = 0

        sim = sim_find.find_similarity(i, bookY, train_book, index_len, devBook+u, devBookY+u)
        bxj = u + user_dev[x] + book_dev[bookY]
        rxj = book_ratings[j, 2]
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

def RMSE(prediction, test, rmse_arr, fold):
    mse = 0
    for i in range(prediction.shape[0]):
        mse = mse + (prediction[i] - test.iloc[i].Rating) ** 2

    rmse = mse / prediction.shape[0]
    rmse = math.sqrt(rmse)
    rmse_arr[fold-1] = rmse
    
    return

def conf_matix(user_dev, book_dev, train_user, train_book, test, u, usl, index_len, metric, fold, rmse_arr):

    prediction = np.zeros(shape=(test.shape[0]), dtype="float")

    for i in range(test.shape[0]):
        print("\tFold: ",fold," pred no: ",i)
        prediction[i] = find_prediction(user_dev, book_dev, train_user, train_book, int(test.iloc[i].User_ID), int(test.iloc[i].Book_ID), u, usl, index_len)

    print("Prediction finished!")

    # th = 7

    precision = np.array([])
    recall = np.array([])
    accuracy = np.array([])

    for j in range(1, 10):
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

        # print("tp:", tp, "tn:", tn, "fp:", fp, "fn:", fn)

        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        accuracy = (tp + tn) / (fp + fn + tp + tn)

        metric[th-1][fold-1][0] = recall
        metric[th-1][fold-1][1] = precision
        metric[th-1][fold-1][2] = accuracy
        print("Metrics finished! th:", j)

    RMSE(prediction, test, rmse_arr, fold)
    print("RMSE Finished!")

    return

def main():
    # kfold.main()

    train_user_arr = []
    train_book_arr = []
    test_arr = []

    usl_arr = []
    bsl_arr = []

    ud_arr = []
    bd_arr = []

    u_arr = []

    for fold in range(1, 11):
        print("Dataset modifying started for fold", fold)

        # dm.minify("train" + str(fold))
        # dm.minify("test" + str(fold))
        #
        # rc.user_rating_average("train" + str(fold))
        # rc.book_rating_average("train" + str(fold))

        train_user = pd.read_csv("./modified-csv/sorted_user_train" + str(fold) + ".csv", sep=",", low_memory=False)
        train_book = pd.read_csv("./modified-csv/sorted_book_train" + str(fold) + ".csv", sep=",", low_memory=False)
        test = pd.read_csv("./modified-csv/sorted_user_test" + str(fold) + ".csv", sep=",", low_memory=False)

        train_user.columns = ["User_ID", "Book_ID", "Rating"]
        train_book.columns = ["User_ID", "Book_ID", "Rating"]
        test.columns = ["User_ID", "Book_ID", "Rating"]

        train_user_arr.append(train_user)
        train_book_arr.append(train_book)
        test_arr.append(test)

        bsl = lb.read_dict("./json-outputs/train" + str(fold) + "-book-start-length.json")
        usl = lb.read_dict("./json-outputs/train" + str(fold) + "-user-start-length.json")

        bsl_arr.append(bsl)
        usl_arr.append(usl)

        book_distinct = list(bsl_arr[fold-1].keys())
        user_distinct = list(usl_arr[fold-1].keys())

        book_distinct = list(map(int, book_distinct))
        user_distinct = list(map(int, user_distinct))

        u = np.average(train_user.iloc[:, 2])
        u_arr.append(u)

        user_dev, book_dev = generate_dev(u, user_distinct, book_distinct, fold)

        bd_arr.append(book_dev)
        ud_arr.append(user_dev)

    print("Data manipulation finished!")

    metric = np.zeros(shape=(10, 10, 3), dtype="float")
    rmse_arr = np.zeros(shape=(10), dtype="float")
    
    for fold in range(1, 11):

        print("Fold started: ", fold)

        # index_len = lb.read_dict("./json-outputs/book-start-length.json")
        
        conf_matix(ud_arr[fold-1], bd_arr[fold-1], train_user_arr[fold-1], train_book_arr[fold-1], test_arr[fold-1], u_arr[fold-1], usl_arr[fold-1], bsl_arr[fold-1], metric, fold, rmse_arr)
        
    avg_rec = []
    avg_prec = []
    avg_acc = []
    
    avg_rmse = np.average(rmse_arr)
    
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

    print("avg_rmse: ", avg_rmse)
    
main()