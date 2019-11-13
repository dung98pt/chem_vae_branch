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
from data_loader import load_data
from model import encode, decode
# Tải dữ liệu 
X_train, X_test, X_val = load_data()

encode = encode()
decode = decode()

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
epochs = 10
for epoch in range(epochs):
  print((epoch+1), "/", epochs)
  vae.fit(X_train, X_train, batch_size= 128, verbose=1, validation_data=(X_val, X_val))
  if (i+1)%1 == 0:
    vae.save_weights("vae_weight_"+(epoch+1)+".h5")
    encode.save_weights("encode_weight_"+(epoch+1)+".h5")
    decode.save_weights("decode_weight_"+(epoch+1)+".h5")
                

# encode.load_weights("encode_weight.h5")
# decode.save_weights("decode_weight.h5")
# VAE.load_weights("vae_weight.h5")


# y = vae.predict(X_test)
# print(y[0][0])