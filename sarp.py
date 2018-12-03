import numpy as np
import pandas as pd
from main.py import u   # Mean of every usr-book pair
from main.py import x   # user index I'm lookin for
from main.py import i   # book index I'm lookin for

sim_matrix = pd.read_cv("./ex.csv")

usr_dev = pd.read_csv("./usr_dev.csv")  # user deviation
bk_dev = pd.read_csv("./book_dev.csv")  # book deviation

bxi = u + usr_dev.iloc[x] + bk_dev.iloc[i]   # düzeltildi
sum = 0

for j in range(0, sim_matrix.shape[0]):
    if not np.isnan(sim_matrix.iloc[i,j]):
        bxj = u + usr_dev.iloc[x] + bk_dev.iloc[j]
        # sum = sum + (sim_matrix.iloc[i,j] * (ratings_matrix.iloc[x,j] - bxj)