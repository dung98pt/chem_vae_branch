%%writefile train.py
from keras.models import Model
from keras import backend as K
from keras.losses import categorical_crossentropy
from keras import objectives
from keras.losses import mse, binary_crossentropy
from keras.utils import to_categorical
import numpy as np
from keras.models import load_model
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from tgru_k2_gpu import TerminalGRU 
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Dropout, MaxPooling1D, Flatten, Dense, BatchNormalization,RepeatVector,GRU,Input,Lambda
from keras.models import Model
from keras import backend as K
from keras.losses import categorical_crossentropy
from keras import objectives
from keras.losses import mse, binary_crossentropy
from keras.utils import to_categorical
import numpy as np
import keras
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from tgru_k2_gpu import TerminalGRU
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Dropout, MaxPooling1D, Flatten, Dense, BatchNormalization,RepeatVector,GRU,Input,Lambda
# chuyển ma trận one-hot 
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
import pandas as pd
link1 = '250k_rndm_zinc_drugs_clean_3.csv'
df1 = pd.read_csv(link1, delimiter=',', names = ['smiles','1','2','3'])
smiles = list(df1.smiles)[1:]
X = []
for smile in smiles:
  try:
    X.append(one_hot(smile[:-1]))  
  except:
      print ("ahihi do ngoc")
X = np.array(X)
print(X.shape)

id = int (X.shape[0])
idx = int (id * 0.8)
idy = int (id * 0.9)
X_train = X[:idx,:,:]
X_val = X[idx:idy,:,:]
X_test = X[idy:id,:,:]
print(X_train.shape)
print(X_test.shape)
  


# Xây model
# latent Z
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

dropout_rate_mid = 0.082832929704794792
def encode():
  inputs = Input(shape=(120,35))
  x = Conv1D(filters=9, kernel_size=9, activation='relu', input_shape=(120,35))(inputs)
  x = BatchNormalization(axis = -1)(x)
  x = Conv1D(filters=9, kernel_size=9, activation='relu')(x)
  x = BatchNormalization(axis = -1)(x)
  x = Conv1D(filters=10, kernel_size=11, activation='relu')(x)
  x = BatchNormalization(axis = -1)(x)
  x = Flatten()(x)
  x = Dense(196, activation='relu')(x)
  x = Dropout(dropout_rate_mid)(x)
  x = BatchNormalization(axis = -1)(x)
  z_mean = Dense(196, name='z_mean')(x)
  z_log_var = Dense(196, name='z_log_var')(x)
  z = Lambda(sampling, output_shape=(196,), name='z')([z_mean, z_log_var])
  model_encode = Model(inputs, [z_mean, z_log_var, z], name = 'encode')
  return model_encode
encode = encode()
encode.summary()

def decode():
  input1 = Input(shape=(120,35))
  input2 = Input(shape=([196]))  
  x = Dense(196, activation='relu')(input2)
  x = Dropout(dropout_rate_mid)(x)
  x = BatchNormalization(axis = -1)(x)
  x = RepeatVector(120)(x)
  x = GRU(448,return_sequences=True, activation='tanh')(x)
  x = GRU(448,return_sequences=True, activation='tanh')(x)
  x = GRU(448,return_sequences=True, activation='tanh')(x)
  x= TerminalGRU(35, rnd_seed=42, recurrent_dropout = 0.0,
    return_sequences=True, activation='softmax',temperature=0.01,
    name='decoder_tgru', implementation=0)([x, input1])
  model_decode = Model([input1, input2], x, name = "decode")
  return model_decode
decode = decode()
decode.summary()



def VAE():
  inputs = Input(shape=(120,35))
  z_mean, z_log_var, z = encode(inputs)
  x = decode([inputs, z])
  VAE = Model(inputs, x, name = "VAE")
  return VAE, z_mean, z_log_var
vae, z_mean, z_log_var  = VAE()
vae.summary() 

''' Test thử 3 hàm loss'''
def vae_loss_binary(x, x_reconstruction):
    xent_loss = K.sum(binary_crossentropy(x, x_reconstruction) , axis = -1)
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return K.mean(xent_loss + kl_loss)

def vae_loss_mse(x, x_reconstruction):
    xent_loss = K.sum(mse(x, x_reconstruction) , axis = -1)
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return K.mean(xent_loss + kl_loss)

def vae_loss_categorical(x, x_reconstruction):
    xent_loss = K.sum(categorical_crossentropy(x, x_reconstruction) , axis = -1)
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    print(kl_loss)
    return K.mean(xent_loss + kl_loss)


optim = optimizers.Adam(lr=0.001, beta_1= 0.97)

loss = [vae_loss_binary, vae_loss_mse, vae_loss_categorical]
metric = ['accuracy', 'categorical_accuracy']

vae.compile(loss = loss[2], optimizer= optim, metrics=[metric[0]])
vae.fit(X_train, X_train, batch_size= 256, epochs=1, verbose=1, validation_data=(X_val, X_val))
                
encode.save("encode.h5")
decode.save("decode.h5")


# encode = load_model("./encode.h5")
# decode = load_model("./decode.h5")
# y = vae.predict(X_test)
# print(y[0][0])