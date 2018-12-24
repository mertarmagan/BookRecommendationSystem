import numpy as np
import pandas as pd
import math
import similarity_finder as sim_find
import labeler as lb
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

def generate_dev(glob_mean, user_distinct, book_distinct, fold):
    books_avg = lb.read_dict("./json-outputs/train" + str(fold) + "-books-average.json")
    users_avg = lb.read_dict("./json-outputs/train" + str(fold) + "-users-average.json")

    # books_avg = lb.read_dict("./json-outputs/extrain-books-average.json")
    # users_avg = lb.read_dict("./json-outputs/extrain-users-average.json")

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

def find_prediction(user_dev, book_dev, train_user, train_book, x, i, u, usl, index_len):
    # x = user, i = book
    
    # print("test user,book:", x, ",", i)
    devUserX = user_dev.get(x, -1)
    if devUserX == -1:
        devUserX = 0

    return devUserX + u
    #### key type check et
    info = index_len.get(str(i), -1)

    if info != -1:
        start = info["start"]
        length = info["length"]
    else:
        start = 0
        length = 0

    user_ratings = train_book.iloc[start:start+length].values

    sum = 0
    total = 0
    # print(user_ratings.shape[0])
    # for y in range(0, user_ratings.shape[0]):
    for y in range(0, user_ratings.shape[0]):
        devUserY = user_dev.get(user_ratings[y, 0], -1)
        if devUserY == -1:
            devUserY = 0

        sim = sim_find.find_similarity_user(x, user_ratings[y, 0], train_user, usl, devUserX+u, devUserY+u)
        # print("user:", x, user_ratings[y, 0], "  sim:", sim)
        # bxj = u + user_dev[x] + book_dev[book_ratings[j,0]]
        ryi = user_ratings[y, 2]
        if sim >= 0:
            # sims = np.append(sims, np.array([[sim, sum]]), axis=0)
            sum = sum + (sim * ryi)
            total = total + sim

    # sims = np.delete(sims, 0, 0)
    # k = int(sims.shape[0] / 2 + 1)
    # # print("k", k)
    # sims = sims[sims[:, 0].argsort()]
    # total = sims[sims.shape[0]-k:sims.shape[0],0]
    # ctr = sims[sims.shape[0]-k:sims.shape[0],1]

    # print("bxi: ", bxi)

    if total != 0:
        rxi = sum/total
    else: # RAPORA YAZILACAK
        rxi = devUserX + u

    # print("user:", str(x), "book:", str(i), " prediction:", rxi)
    return rxi

def RMSE(prediction, test, rmse_arr, fold):
    mse = 0

    for i in range(prediction.shape[0]):
        mse = mse + (prediction[i] - test.iloc[i].Rating) ** 2

    rmse = math.sqrt(mse)
    rmse = rmse / prediction.shape[0]
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

    # Testing for every threshold value from 1 to 9
    for j in range(1,10):
        tp = 0
        tn = 0
        fp = 0
        fn = 0

        th = j
        print("\tThreshold:", th)
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
    print("RMSE Finished for Fold", fold, "!")

    return

def main():

    # sim_find.sample_generator2()

    # dm.sortDatasetBook("extrain")
    # dm.findIndexLengthBooks("extrain")
    # dm.sortDatasetUser("extrain")
    # dm.findIndexLengthUsers("extrain")

    # rc.user_rating_average("extrain")
    # rc.book_rating_average("extrain")

    # train_user = pd.read_csv("./modified-csv/sorted_user_extrain.csv", sep=",", low_memory=False)
    # train_book = pd.read_csv("./modified-csv/sorted_book_extrain.csv", sep=",", low_memory=False)
    # test = pd.read_csv("./modified-csv/extest.csv", sep=",", low_memory=False)

    # train_user.columns = ["User_ID", "Book_ID", "Rating"]
    # train_book.columns = ["User_ID", "Book_ID", "Rating"]
    # test.columns = ["User_ID", "Book_ID", "Rating"]
        
    # bsl = lb.read_dict("./json-outputs/extrain-book-start-length.json")
    # usl = lb.read_dict("./json-outputs/extrain-user-start-length.json")

    # book_distinct = list(bsl.keys())
    # user_distinct = list(usl.keys())

    # book_distinct = list(map(int, book_distinct))
    # user_distinct = list(map(int, user_distinct))

    # u = np.average(train_user.iloc[:, 2])
    
    # # print(u)

    # # ITEM ITEM da CHECK ET
    # user_dev, book_dev = generate_dev(u, user_distinct, book_distinct, 1)

    # # print("book dev:", book_dev)
    # # print("user dev:", user_dev)

    # metric = np.zeros(shape=(10, 10, 3), dtype="float")
    # rmse_arr = np.zeros(shape=(10), dtype="float")

    # conf_matix(user_dev, book_dev, train_user, train_book, test, u, usl, bsl, metric, 1, rmse_arr)

    # print("RMSE:", rmse_arr[0])
    # print("metric:", metric[0])

    ###########################################################################################################
    # kfold.main()

    train_user_arr = []
    train_book_arr = []
    test_arr = []

    usl_arr = []
    bsl_arr = []

    ud_arr = []
    bd_arr = []

    u_arr = []

    # range değişti !!!!!
    for fold in range(1, 2):
        print("Dataset modifying started for fold", fold)

        # dm.minify("train" + str(fold))
        # dm.minify("test" + str(fold))

        # rc.user_rating_average("train" + str(fold))
        # rc.book_rating_average("train" + str(fold))

        train_user = pd.read_csv("./modified-csv/sorted_user_train" + str(fold) + ".csv", sep=",", low_memory=False)
        train_book = pd.read_csv("./modified-csv/sorted_book_train" + str(fold) + ".csv", sep=",", low_memory=False)
        test = pd.read_csv("./modified-csv/sorted_user_test" + str(fold) + ".csv", sep=",", low_memory=False)

        # değişti !!!!!
        test = test.iloc[:2000]

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
    rmse_arr = np.zeros(shape=(1), dtype="float")

    # print("SAAAA:", len(test_arr[0]))
    # TODO? minimized range for one fold
    for fold in range(1, 2):

        print("Fold started: ", fold)

        # index_len = lb.read_dict("./json-outputs/book-start-length.json")
        
        conf_matix(ud_arr[fold-1], bd_arr[fold-1], train_user_arr[fold-1], train_book_arr[fold-1], test_arr[fold-1], u_arr[fold-1], usl_arr[fold-1], bsl_arr[fold-1], metric, fold, rmse_arr)
        
    avg_rec = []
    avg_prec = []
    avg_acc = []
    
    avg_rmse =  np.average(rmse_arr)
    
    for i in range(10):
        
        sum_rec = 0
        sum_prec = 0
        sum_acc = 0
        
        for j in range(10):
            
            sum_rec += metric[i][j][0]
            sum_prec += metric[i][j][1]
            sum_acc += metric[i][j][2]
        
        avg_rec.append(sum_rec/1)
        avg_prec.append(sum_prec/1)
        avg_acc.append(sum_acc/1)
    # DEGİSTİ (bölüler)

    print("avg_rec: ", avg_rec)
    print("avg_prec: ", avg_prec)
    print("avg_acc: ", avg_acc)

    print("avg_rmse: ", avg_rmse)
    


main()