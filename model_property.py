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
from data_loader import load_data_smiles, load_data_propertys
from keras.losses import mse

# load data đầu vào là ma trận one-hot, đầu ra là 3 thuộc tính 
# X_train, X_val, X_test = load_data_smiles()
# pr_train, pr_val, pr_test = load_data_propertys()

# mô hình nhánh phụ 
encode = encode()
encode.load_weights("encode_weights.h5")

inputs = Input(shape=(120,35))
z_mean, z_log_var, z = encode(inputs)
x = Dense(196, activation='relu')(z)
x = Dense(196, activation='relu')(x)
x = Dense(196, activation='relu')(x)
x = Dense(3, activation='tanh')(x)
model_prt = Model(inputs,x, name = 'model_property')
model_prt.summary()

# đóng băng encode 
for layer in encoder.layers:
    layer.trainable = False

# optim = optimizers.Adam()
# model_prt.compile(loss = 'mse', optimizer= optim, metrics=['accuracy'])

# for epoch in range(50):
#   values =[]
#   print((epoch+1), "/", 50)
#   history= model_prt.fit(X_train, pr_train, batch_size= 128, verbose=1, validation_data=(X_val, pr_val))
#   values.append(history.history['acc'])
#   values.append(history.history['loss'])
#   values.append(history.history['val_acc'])
#   values.append(history.history['val_loss'])
#   values = np.array(values)
#   # np.save(path+'epoch'+str(epoch+1)+'.npy', values)
#   if history.history['acc'][0]>0.7:
#     vae.save_weights(path+"vae/vae_weight_mse_"+str(epoch+1)+".h5")
#     encode.save_weights(path+"encode/encode_weight_mse_"+str(epoch+1)+".h5")
#     decode.save_weights(path+"decode/decode_weight_mse_"+str(epoch+1)+".h5")

'''training model'''
