# -*- coding: utf-8 -*-
import numpy as np
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.models import Sequential, load_model
from keras.callbacks import Callback
import tensorflow as tf
import  pandas as pd
import  os
import  keras.callbacks
from sklearn.model_selection import train_test_split
import tensorflow.compat.v1.keras.backend as KTF
from sklearn.metrics import mean_absolute_error,mean_squared_error
from math import sqrt
import tensorflow_model_optimization as tfmot
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn import preprocessing
from math import sqrt

#设定为自增长
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.compat.v1.Session(config=config)
KTF.set_session(session)

start1=97
end1=104
TestLength = (end1 -start1)*48+1

#start1 = 84
#end1 = 98

#start1 = 24
#end1 = 38


f1d=open(r'C:\\Users\\310\\Desktop\\陈志鹏程序\\陶一凡实验\\1213high schools\\Primary Schools\\Creigiau Primary School\\places.857_2013_elec.csv',encoding= "utf-8")
df=pd.read_csv(f1d)     #读入数据
data1d = df.iloc[start1:end1,1:50]
f1q=open(r'C:\\Users\\310\\Desktop\\陈志鹏程序\\陶一凡实验\\1213high schools\\Primary Schools\Creigiau Primary School\\places.857_2013_gas.csv',encoding= "utf-8")
df=pd.read_csv(f1q)     #读入数据
data1q = df.iloc[start1:end1,1:50]
data1d = data1d.T
data1q = data1q.T
df1dd = data1d
df1d=pd.concat(df1dd.iloc[:,i] for i in range(df1dd.shape[1]))
df1d.index=np.arange(len(df1d))



df1qq = data1q

df1q=pd.concat(df1qq.iloc[:,i] for i in range(df1qq.shape[1]))
df1q.index=np.arange(len(df1q))


data = pd.DataFrame([df1q,df1d])
data = data.T
data = np.array(data)
data.shape
TrainLength = len(data)
def slidingwindow(data,timesteps):
    train  = []
    for i in range(0, data.shape[0]-timesteps+1):

        data1 = data[i:i+timesteps,:]
        train.append(data1)


    return np.array(train)

min_max_scalerTrain = preprocessing.MinMaxScaler()

dataScaler = min_max_scalerTrain.fit_transform(data)#归一化后的数据

data_X = dataScaler[0:TrainLength]

train_X = slidingwindow(data_X,48)#滑动窗口后的数据
train_X = train_X.reshape(TrainLength-47,48,2)

train_C = np.array(dataScaler[47:TrainLength+1,0],dtype='float64').reshape(TrainLength-47,1)
train_H = np.array(dataScaler[47:TrainLength+1,1],dtype='float64').reshape(TrainLength-47,1)



model = load_model("C:\\Users\\310\\Desktop\\陈志鹏程序\\陶一凡实验\\Greenhill Primary School\\ceshi64.h5")
model.layers[0].trainable = False
model.layers[1].trainable = False
model.layers[2].trainable = False
model.layers[3].trainable = False
model.layers[4].trainable = False
model.layers[5].trainable = False
model.layers[6].trainable = False
model.layers[7].trainable = False
model.layers[8].trainable = True

#进行测试
#model.layers[0].trainable = False
print("-------------------------------------------------")
model.summary()
for layer in model.layers:
    print(layer.name, ' is trainable? ', layer.trainable)
    #model.layers[0].trainable = False
    

model.fit(train_X,[train_H,train_C],epochs=75,batch_size=21)
model.save("./weitiao64.h5")















