# -*- coding: utf-8 -*-
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
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow.compat.v1.keras.backend as KTF
import tensorflow_model_optimization as tfmot
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn import preprocessing
from math import sqrt

start1 = 97
end1 = 104+7
TrainLength = 335
TestLength = 672+7*48*0

f1d=open(r'C:\\Users\\310\\Desktop\\陈志鹏程序\\陶一凡实验\\1213high schools\\Primary Schools\\Creigiau Primary School\\places.857_2013_elec.csv',encoding= "utf-8")
df=pd.read_csv(f1d)     #读入数据
data_1d=df.iloc[:7,1:50]
data1d = df.iloc[start1:end1,1:50]

f1q=open(r'C:\\Users\\310\\Desktop\\陈志鹏程序\\陶一凡实验\\1213high schools\\Primary Schools\\Creigiau Primary School\\places.857_2013_gas.csv',encoding= "utf-8")
df=pd.read_csv(f1q)     #读入数据
data_1q=df.iloc[:7,1:50]
data1q = df.iloc[start1:end1,1:50]

data1d = data1d.T
data1q = data1q.T

#data11.plot()
#plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
#plt.show()

df1dd = data1d

df1d=pd.concat(df1dd.iloc[:,i] for i in range(df1dd.shape[1]))
df1d.index=np.arange(len(df1d))
#print(df1d)


df1qq = data1q

df1q=pd.concat(df1qq.iloc[:,i] for i in range(df1qq.shape[1]))
df1q.index=np.arange(len(df1q))




data = pd.DataFrame([df1q,df1d])
data = data.T
data = np.array(data)

def slidingwindow(data,timesteps):
    train  = []
    for i in range(0, data.shape[0]-timesteps+1):

        data1 = data[i:i+timesteps,:]
        train.append(data1)


    return np.array(train)

#训练数据处理
min_max_scalerTrain = preprocessing.MinMaxScaler()
min_max_scalerTest = preprocessing.MinMaxScaler()
min_max_scalerTestC = preprocessing.MinMaxScaler()
min_max_scalerTestH = preprocessing.MinMaxScaler()
#min_max_scalerTestE = preprocessing.MinMaxScaler()

test_X = np.array(data[TrainLength-47+7*48*0:TestLength],dtype='float64')
test_C = min_max_scalerTestC.fit_transform(test_X[:,0:1])
test_H = min_max_scalerTestH.fit_transform(test_X[:,1:2])
#test_E = min_max_scalerTestE.fit_transform(test_X[:,2:3])

test_X = min_max_scalerTest.fit_transform(test_X)
test_X = slidingwindow(test_X,48)
test_X = test_X.reshape(TestLength-TrainLength-7*48*0,48,2)

verificationC = np.array(data[TrainLength+7*48*0:TestLength,0])
verificationH = np.array(data[TrainLength+7*48*0:TestLength,1])

#test_X,test_Y = create_dataset(testdata,336,48)
#test_X.shape
model = load_model("C:\\Users\\310\\Desktop\\陈志鹏程序\\陶一凡实验\\Greenhill Primary School\\weitiao64.h5")

y_hat = np.array(model.predict(test_X))
#重组


#反归一化
y_hat_D = min_max_scalerTestH.inverse_transform(y_hat[0]).reshape(TestLength-TrainLength-7*48*0,)
y_hat_Q = min_max_scalerTestC.inverse_transform(y_hat[1]).reshape(TestLength-TrainLength-7*48*0,)


plt.plot(y_hat_Q,color='r',label='Overall')
plt.plot(verificationC,color='b',label='actual',)
font1 = {'family' : 'Times New Roman',

'weight' : 'normal',

'size'   : 10.5,

}
plt.legend(loc=1,  prop=font1)
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.xlabel('滑动窗口索引')
plt.ylabel('负荷/千瓦时')
plt.show()





plt.plot(y_hat_D,color='b',label='Overall',)
plt.plot(verificationH,color='g',label='actual',)
font1 = {'family' : 'Times New Roman',

'weight' : 'normal',

'size'   : 10.5,

}
plt.legend(loc=1,  prop=font1)
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.xlabel('滑动窗口索引')
plt.ylabel('负荷/千瓦时')
plt.show()

def mape(y_true, y_pred):
    """
    参数:
    y_true -- 测试集目标真实值
    y_pred -- 测试集目标预测值

    返回:
    mape -- MAPE 评价指标
    """
    
    n = len(y_true)
    mape = sum(np.abs((y_true - y_pred)/y_true))/n*100
    return mape

mapeq = mape(verificationC,y_hat_Q)
maeq = mean_absolute_error(verificationC,y_hat_Q)
mseq = mean_squared_error(verificationC,y_hat_Q)
rmseq = sqrt(mean_squared_error(verificationC,y_hat_Q))
print('Test RMSE: %.3f' % rmseq)
print('Test MAE: %.3f' % maeq)
#print('Test MAPE: %.3f' % mapeq)
print('Test MSE: %.3f' % mseq)


maped = mape(verificationH,y_hat_D)
maed = mean_absolute_error(verificationH,y_hat_D)
msed = mean_squared_error(verificationH,y_hat_D)
rmsed = sqrt(mean_squared_error(verificationH,y_hat_D))
print('Test RMSE: %.3f' % rmsed)
print('Test MAE: %.3f' % maed)
#print('Test MAPE: %.3f' % maped)
print('Test MSE: %.3f' % msed)






