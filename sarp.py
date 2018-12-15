import numpy as np
import pandas as pd
import math
import similarity_finder as sim_find
import labeler as lb
# from main.py import u   # Mean of every usr-book pair
# from main.py import x   # user index I'm lookin for
# from main.py import i   # book index I'm lookin for
# from .similarity_finder import similarity_finder

# sim_matrix = pd.read_cv("./ex.csv")

# usr_dev = pd.read_csv("./usr_dev.csv")  # user deviation
# bk_dev = pd.read_csv("./book_dev.csv")  # book deviation

def generate_dev(df, glob_mean, user_distinct, book_distinct):
    books_avg = lb.read_dict("./json-outputs/books-average.json")
    users_avg = lb.read_dict("./json-outputs/users-average.json")

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

def find_prediction(user_distinct, book_distinct, user_dev, book_dev, train, x, i, u, index_len):
    
    bxi = u + user_dev[x] + book_dev[i]
    
    # print("bxi: ", bxi)
    sum = 0
    total = 0

    sims = np.array([[ None, None]])

    for item in book_distinct:
        if item != i:
            for index, user in train.iterrows():
                if user["Book_ID"] == item and user["User_ID"] == x:
                    # print(str(i), str(item), index_len)
                    sim = sim_find.find_similarity(str(i), str(item), train, index_len)
                    
                    bxj = u + user_dev[x] + book_dev[item]
                                        
                    rxj = 0
                    for index, row in train.iterrows():
                        if row["User_ID"] == user["User_ID"] and row["Book_ID"] == item:
                            rxj = row["Rating"]
                            # print("user:", x, "book:", row["Book_ID"], "rxj: ", rxj)

                    sum = sum + (sim * (rxj - bxj))
                    # print("sim: ",sim, " sum: ", sum)
                    if sim > 0:
                        sims = np.append(sims, np.array([[sim, sum]]), axis=0)
                    # print("sim * (rxj -bxj)", "x", x, "j", item)

        sum = 0

    sims = np.delete(sims, 0, 0)
    k = int(sims.shape[0] / 2 + 1) 
    # print("k", k)
    sims = sims[sims[:, 0].argsort()]
    total = sims[sims.shape[0]-k:sims.shape[0],0]
    ctr = sims[sims.shape[0]-k:sims.shape[0],1]
    # print("np.sum(ctr)", np.sum(ctr))
    # print("np.sum(total)", np.sum(total))
    
    rxi = bxi
    if np.sum(total) != 0:
        rxi = rxi + np.sum(ctr) / np.sum(total)
    print(rxi)
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

def conf_matix(user_distinct, book_distinct, user_dev, book_dev, train, test, u, index_len):

    prediction = np.zeros(shape=(test.shape[0]), dtype="float")

    for i in range(test.shape[0]):
        prediction[i] = find_prediction(user_distinct, book_distinct, user_dev, book_dev, train, test[i, 0], test[i, 1], u, index_len)

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for i in range(prediction.shape[0]):
        if prediction[i] >= 6:
            if test[i,2] >= 6:
                tp = tp + 1
                # print("prediction:", prediction[i], "value:", test[i, 2], "user:", test[i, 0], "book:", test[i, 1])
            else:
                # print("prediction:", prediction[i], "value:", test[i, 2], "user:", test[i,0], "book:", test[i,1])
                fp = fp + 1
        else:
            if test[i,2] < 6:
                tn = tn + 1
            else:
                fn = fn + 1

    print("tp:", tp, "tn:", tn, "fp:", fp, "fn:", fn)

def main():
    # train = pd.read_csv("./ex_similarity/sorted_train.csv", sep=",", low_memory=False)

    train = pd.read_csv("./partition/train1.csv", sep=",", low_memory=False)
    test = pd.read_csv("./partition/test1.csv", sep=",", low_memory=False)

    train.columns = ["User_ID", "Book_ID", "Rating"]
    test.columns = ["User_ID", "Book_ID", "Rating"]

    bsl = lb.read_dict("./json-outputs/book-start-length.json")
    usl = lb.read_dict("./json-outputs/user-start-length.json")

    book_distinct = list(bsl.keys())
    user_distinct = list(usl.keys())

    book_distinct = list(map(int, book_distinct))
    user_distinct = list(map(int, user_distinct))
    print("Distinct job finished.")

    u = np.average(train.iloc[:, 2])

    # print(generate_dev(train, u))

    book_dev, user_dev = generate_dev(train, u, user_distinct, book_distinct)

    print("Generate dev finished.")

    # find_prediction(user_distinct, book_distinct, user_dev, book_dev, train, 12, 5, u)

    index_len = lb.read_dict("./json-outputs/book-start-length.json")
    # print(index_len)
    # conf_matix(user_distinct, book_distinct, user_dev, book_dev, train, np.copy(train), u, index_len)
    conf_matix(user_distinct, book_distinct, user_dev, book_dev, train, test, u, index_len)

    # print(u)

    # print(find_similarity(train, 1, 2))
    # find_prediction()
    # similarity_finder()
    # generate_devs()
    # find_prediction()

main()