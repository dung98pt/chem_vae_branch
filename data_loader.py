from keras.utils import to_categorical
import pandas as pd
from sklearn.model_selection import train_test_split as stt
from sklearn import preprocessing
import numpy as np



DICT = {'5': 29, '=': 22, 'N': 31, 'l': 16, 'H': 18, ']': 3, '@': 21, '6': 1, 'O': 17, 'c': 19, '2': 27, '8': 25, '3': 4, '7': 0, 'I': 15, 'C': 26, 'F': 28, '-': 7, 'P': 24, '/': 9, ')': 13, ' ': 34, '#': 14, 'r': 30, '\\': 33, '1': 20, 'n': 23, '+': 32, '[': 12, 'o': 2, 's': 5, '4': 11, 'S': 8, '(': 6, 'B': 10}

# hàm chuyển one-hot chuyển one-hot 
def one_hot(str, LEN_MAX = 120):
    str = list(str)
    if len(str) < LEN_MAX:
        for i in range(LEN_MAX - len(str)):
            str.append(" ")
    hot  = []
    for char in list(str):
      hot.append(DICT[char])
    return to_categorical(hot)

'''
chuyển smiles thành ma trận one-hot và lưu dưới dạng mảng là file .npy
SMILES.npy chưa ma trận one-hot
propertys.npy chứa 3 thuộc tính 'logP','qed','SAS'
'''
def data_to_array():
  link1 = '250k_rndm_zinc_drugs_clean_3.csv'
  df1 = pd.read_csv(link1, delimiter=',', names = ['smiles','logP','qed','SAS'])
  smiles = list(df1.smiles)[1:]
  logP = list(df1.logP)[1:]
  logP = np.array(logP).reshape(len(logP))
  qed = list(df1.qed)[1:]
  qed = np.array(qed).reshape(len(qed))
  SAS = list(df1.SAS)[1:]
  SAS = np.array(SAS).reshape(len(SAS))
  X = []	
  for smile in smiles[:]:
  	try:
  		X.append(one_hot(smile[:-1]))  
  	except:
  		pass
  propertys = np.concatenate([logP, qed, SAS], axis = 1)
  np.save('./SMILES.npy',X)
  np.save('./propertys.npy', propertys)

'''Chia dữ liệu thành 3 tập train, test, val với tỉ lệ 8,1,1'''
def load_data_smiles(smiles = np.load('./SMILES.npy')):
	X_train, X_test = stt(smiles, test_size=0.1, random_state=42)
	X_train, X_val = stt(X_train, test_size=0.111, random_state=42)
	return X_train, X_val, X_test 

def load_data_propertys(propertys = np.load('./propertys.npy')):
	pp = preprocessing.MinMaxScaler() # chuẩn hóa dữ liệu về khoảng -1,1
	propertys = mms.fit_transform(propertys)
	X_train, X_test = stt(propertys, test_size=0.1, random_state=42)
	X_train, X_val = stt(X_train, test_size=0.111, random_state=42)
	return X_train, X_val, X_test 


