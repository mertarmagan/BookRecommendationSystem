import numpy as np
import pandas as pd
# from main.py import u   # Mean of every usr-book pair
# from main.py import x   # user index I'm lookin for
# from main.py import i   # book index I'm lookin for
# from .similarity_finder import similarity_finder

# import similarity_finder

# sim_matrix = pd.read_cv("./ex.csv")

# usr_dev = pd.read_csv("./usr_dev.csv")  # user deviation
# bk_dev = pd.read_csv("./book_dev.csv")  # book deviation



def generate_dev(df, u, user_distinct, book_distinct):

    x_dict = {}
    y_dict = {}

    x_arr = []
    y_arr = []

    book_dev = {}
    user_dev = {}

    x_yarr = {}

    ######## compress it all #########

    # print(user_distinct)
    # print("-------------")
    # print(book_distinct)

    for item in book_distinct:
        sum = 0
        x_dict = {}
        for index, row in df.iterrows():
            if row["ISBN"] == item:
                x_dict[row["User-ID"]] = row["Book-Rating"]
                temp = row["User-ID"]
                sum = sum + row["Book-Rating"]
        book_dev[item] = sum/len(x_dict) - u
        x_yarr[item] = sum/len(x_dict)
        # print(x_dict, sum/len(x_dict))

    # print(u)
    # print(book_dev)
    # print(x_yarr)

    for item in user_distinct:
        sum = 0
        y_dict = {}
        for index, row in df.iterrows():
            if row["User-ID"] == item:
                y_dict[row["ISBN"]] = row["Book-Rating"]
                sum = sum + row["Book-Rating"]
        user_dev[item] = sum/len(y_dict) - u
    
    # print(user_dev)

    # print("book-rating by user", y_arr)
    # print("+++++++++")
    # print("book-rating by user", x_dict)

    return user_dev, book_dev

def find_mean(_dict):
    sum = 0
    for i in _dict:  # iterate over keys
        sum += _dict[i]

    mean = sum/len(_dict)

    return mean

def find_prediction(user_distinct, book_distinct, user_dev, book_dev, train, x, i, u):    
    
    bxi = u + user_dev[x] + book_dev[i]
    
    # print("bxi: ", bxi)
    sum = 0
    total = 0

    sims = np.array([[ None, None]])

    for item in book_distinct:
        if item != i:
            for index, user in train.iterrows():
                if user["ISBN"] == item and user["User-ID"] == x:
                    sim = find_similarity(train, i, item)
                    
                    bxj = u + user_dev[x] + book_dev[item]
                                        
                    rxj = 0
                    for index, row in train.iterrows():
                        if row["User-ID"] == user["User-ID"] and row["ISBN"] == item:
                            rxj = row["Book-Rating"]
                            # print("user:", x, "book:", row["ISBN"], "rxj: ", rxj)

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
    # print(rxi)
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

def conf_matix(user_distinct, book_distinct, user_dev, book_dev, train, test, u):

    prediction = np.zeros(shape=(test.shape[0]), dtype="float")

    for i in range(test.shape[0]):
        prediction[i] = find_prediction(user_distinct, book_distinct, user_dev, book_dev, train, test[i, 0], test[i, 1], u)

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for i in range(prediction.shape[0]):
        if prediction[i] >= 3:
            if test[i,2] >= 3:
                tp = tp + 1
            else:
                print("prediction:", prediction[i], "value:", test[i, 2], "user:", test[i,0], "book:", test[i,1])
                fp = fp + 1
        else:
            if test[i,2] < 3:
                tn = tn + 1
            else:
                fn = fn + 1

    print("tp:", tp, "tn:", tn, "fp:", fp, "fn:", fn)

def find_similarity(df, x, y):
    x_dict = {}
    y_dict = {}

    for index, row in df.iterrows():
        if row["ISBN"] == x:
            x_dict[row["User-ID"]] = row["Book-Rating"]

    for index, row in df.iterrows():
        if row["ISBN"] == y:
            y_dict[row["User-ID"]] = row["Book-Rating"]

    # print(x_dict)
    # print(y_dict)

    res_x_dict = {}
    res_y_dict = {}
    # Finding the intersecting users
    for item in x_dict.keys():
        if item in y_dict:
            res_y_dict[item] = y_dict[item]
            res_x_dict[item] = x_dict[item]

    # print(res_x_dict)
    # print(res_y_dict)

    mean_x = find_mean(x_dict)
    mean_y = find_mean(y_dict)

    # print(mean_x, mean_y)

    for i in x_dict:
        val = x_dict[i]
        x_dict[i] = val - mean_x

    for i in y_dict:
        val = y_dict[i]
        y_dict[i] = val - mean_y

    # print(x_dict)
    # print(y_dict)

    sum = 0
    sum_x = 0
    sum_y = 0

    for i in x_dict:
        sum_x += x_dict[i] ** 2

    for i in y_dict:
        sum_y += y_dict[i] ** 2

    for i in res_x_dict:
        sum += x_dict[i] * y_dict[i]

    sum_x = sum_x ** (1/2)
    sum_y = sum_y ** (1/2)

    # if sum_x == 0:
    #     sum_x = 1
    #
    # if sum_y == 0:
    #     sum_y = 1

    sim = sum / (sum_x * sum_y)
    return sim

def main():
    train = pd.read_csv("./ex_similarity/train.csv", sep=",", low_memory=False)

    book_distinct = np.array([], dtype="int")
    user_distinct = np.array([], dtype="int")

    user_distinct = np.append(user_distinct, [train.iloc[0, 0]])
    book_distinct = np.append(book_distinct, [train.iloc[0, 1]])

    for i in range(0, train.shape[0]):
        if train.iloc[i, 0] not in user_distinct:
            user_distinct = np.append(user_distinct, [train.iloc[i, 0]])
        if train.iloc[i, 1] not in book_distinct:
            book_distinct = np.append(book_distinct, [train.iloc[i, 1]])
    
    book_distinct = np.sort(book_distinct)

    u = np.average(train.iloc[:, 2])

    # print(generate_dev(train, u))

    dev = generate_dev(train, u, user_distinct, book_distinct)
    book_dev = dev[1]
    user_dev = dev[0]

    # find_prediction(user_distinct, book_distinct, user_dev, book_dev, train, 12, 5, u)

    conf_matix(user_distinct, book_distinct, user_dev, book_dev, train, np.copy(train), u)

    # print(u)

    # print(find_similarity(train, 1, 2))
    # find_prediction()
    # similarity_finder()
    # generate_devs()
    # find_prediction()

main()