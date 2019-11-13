from keras.utils import to_categorical
import pandas as pd
from sklearn.model_selection import train_test_split as stt
import numpy as np



DICT = {'5': 29, '=': 22, 'N': 31, 'l': 16, 'H': 18, ']': 3, '@': 21, '6': 1, 'O': 17, 'c': 19, '2': 27, '8': 25, '3': 4, '7': 0, 'I': 15, 'C': 26, 'F': 28, '-': 7, 'P': 24, '/': 9, ')': 13, ' ': 34, '#': 14, 'r': 30, '\\': 33, '1': 20, 'n': 23, '+': 32, '[': 12, 'o': 2, 's': 5, '4': 11, 'S': 8, '(': 6, 'B': 10}
def one_hot(str, LEN_MAX = 120):
    str = list(str)
    if len(str) < LEN_MAX:
        for i in range(LEN_MAX - len(str)):
            str.append(" ")
    hot  = []
    for char in list(str):
      hot.append(DICT[char])
    return to_categorical(hot)

# đọc smiles vô rồi chuyển one-hot 

def load_data():
	link1 = '250k_rndm_zinc_drugs_clean_3.csv'
	df1 = pd.read_csv(link1, delimiter=',', names = ['smiles','1','2','3'])
	smiles = list(df1.smiles)[1:]
	X = []
	for smile in smiles[:1000]:
		try:
			X.append(one_hot(smile[:-1]))  
		except:
			pass

	X = np.array(X)
	print(X.shape)
	X_train, X_test = stt(X, test_size=0.1, random_state=42)
	X_train, X_val = stt(X_train, test_size=0.11, random_state=42)
	return X_train, X_val, X_test 



