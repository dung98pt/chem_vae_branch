from keras.models import Model
from keras import backend as K
from keras.utils import to_categorical
import numpy as np
from keras.models import load_model
from keras import optimizers
from tgru_k2_gpu import TerminalGRU 
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Dropout, MaxPooling1D, Flatten, Dense, BatchNormalization,RepeatVector,GRU,Input,Lambda
from model import encode, decode

# load data


# buid model
encoder = encode()
encoder.load_weights("encode_weight.h5")
print(len(encoder.layers))
inputs = Input(shape=(120,35))
z_mean, z_log_var, z = encoder(inputs)
x = Dense(196, activation='relu')(z)
x = Dense(196, activation='relu')(x)
x = Dense(196, activation='relu')(x)
x = Dense(3)(x)
model_prt = Model(inputs,x)
model_prt.summary()

# for layer in encoder.layers:
#     layer.trainable = False
for layer in encoder.layers[:-3]:
	print(layer)


