from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.losses import mse, binary_crossentropy
from keras.utils import to_categorical
import numpy as np
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from tgru_k2_gpu import TerminalGRU 
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Dropout, MaxPooling1D, Flatten, Dense, BatchNormalization,RepeatVector,GRU,Input,Lambda

# latent Z
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

dropout_rate_mid = 0.1

def encode():
  inputs = Input(shape=([120,35]))
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
