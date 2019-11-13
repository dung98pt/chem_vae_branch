from keras.utils import to_categorical
import numpy as np
import os
# chuyển ma trận one-hot 
# DICT = {'5': 29, '=': 22, 'N': 31, 'l': 16, 'H': 18, ']': 3, '@': 21, '6': 1, 'O': 17, 'c': 19, '2': 27, '8': 25, '3': 4, '7': 0, 'I': 15, 'C': 26, 'F': 28, '-': 7, 'P': 24, '/': 9, ')': 13, ' ': 34, '#': 14, 'r': 30, '\\': 33, '1': 20, 'n': 23, '+': 32, '[': 12, 'o': 2, 's': 5, '4': 11, 'S': 8, '(': 6, 'B': 10}
# def one_hot(str, LEN_MAX = 120):
#     str = list(str)
#     if len(str) < LEN_MAX:
#         for i in range(LEN_MAX - len(str)):
#             str.append(" ")
#     hot  = []
#     for char in list(str):
#       hot.append(DICT[char])
#     return to_categorical(hot)

# # đọc smiles vô rồi chuyển one-hot 
# import pandas as pd
# link1 = '250k_rndm_zinc_drugs_clean_3.csv'
# df1 = pd.read_csv(link1, delimiter=',', names = ['smiles','1','2','3'])
# smiles = list(df1.smiles)[1:]
# idx = int (len(smiles) /2000)

# for i in range(idx):
#   X = []
#   for smile in smiles[i*2000:(i+1)*2000]:
#     try:
#       X.append(one_hot(smile[:-1]))  
#     except:
#         print ("ahihi do ngoc")
#   X = np.array(X)
#   np.save('./data/'+str(i)+'.npy', X)
#
# X = []
# for smile in smiles[idx*2000:]:
#   try:
#     X.append(one_hot(smile[:-1]))  
#   except:
#     print ("ahihi do ngoc")
# X = np.array(X)
# np.save('./data/'+'end'+'.npy', X)

dir1 = './data/'
# dir2 = './data/89.npy'

# a = np.load(dir1)
# b = np.load(dir2)
# print(a.shape)
# print(b.shape)
X = np.load('./end.npy')
print(X.shape)
X = np.dtype(X, np.int8)
np.save('./data.npy', X)