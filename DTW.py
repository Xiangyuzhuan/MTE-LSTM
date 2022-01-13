# -*- coding: utf-8 -*-
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
from scipy.io import loadmat
from sklearn.metrics import mean_absolute_error,mean_squared_error
#from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model, Model
from keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
from keras.initializers import glorot_uniform
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras import backend as K
from keras.models import Sequential
import matplotlib.pyplot as plt
from numpy import concatenate
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib as mpl
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
import multiprocessing
from sklearn.metrics import silhouette_score
import math

from sklearn.decomposition import PCA
from pandas.core.frame import DataFrame


start1=20
end1=104#学校1 2  86


f1d=open(r'C:\Users\310\Desktop\MTE-LSTM\1213high schools\Primary Schools\Greenhill School\places.874_2013_elec.csv',encoding= "utf-8")
df=pd.read_csv(f1d)     #读入数据
data_1d=df.iloc[:7,1:50]
data1d = df.iloc[end1-14:end1,1:50]

f1q=open(r'C:\Users\310\Desktop\MTE-LSTM\1213high schools\Primary Schools\Greenhill School\places.874_2013_gas.csv',encoding= "utf-8")
df=pd.read_csv(f1q)     #读入数据
data_1q=df.iloc[:7,1:50]
data1q = df.iloc[end1-14:end1,1:50]
data1d = data1d.T
data1q = data1q.T
df1dd = data1d
df1d=pd.concat(df1dd.iloc[:,i] for i in range(df1dd.shape[1]))
df1d.index=np.arange(len(df1d))
df1qq = data1q
df1q=pd.concat(df1qq.iloc[:,i] for i in range(df1qq.shape[1]))
df1q.index=np.arange(len(df1q))
data1 = pd.DataFrame([df1q,df1d])
data1 = data1.T
data1 = np.array(data1)


f2d=open(r'C:\Users\310\Desktop\MTE-LSTM\1213high schools\Primary Schools\Birchgrove Primary School\places.833_2013_elec.csv',encoding= "utf-8")
df=pd.read_csv(f2d)     #读入数据
data_2d=df.iloc[:7,1:50]
data2d = df.iloc[start1:end1,1:50]

f2q=open(r'C:\Users\310\Desktop\MTE-LSTM\1213high schools\Primary Schools\Birchgrove Primary School\places.833_2013_gas.csv',encoding= "utf-8")
df=pd.read_csv(f2q)     #读入数据
data_2q=df.iloc[:7,1:50]
data2q = df.iloc[start1:end1,1:50]
data2d = data2d.T
data2q = data2q.T
df2dd = data2d
df2d=pd.concat(df2dd.iloc[:,i] for i in range(df2dd.shape[1]))
df2d.index=np.arange(len(df2d))
df2qq = data2q
df2q=pd.concat(df2qq.iloc[:,i] for i in range(df2qq.shape[1]))
df2q.index=np.arange(len(df2q))
data2 = pd.DataFrame([df2q,df2d])
data2 = data2.T
data2 = np.array(data2)
data2.shape
data2


f3d=open(r'C:\Users\310\Desktop\MTE-LSTM\1213high schools\Primary Schools\Creigiau Primary School\places.857_2013_elec.csv',encoding= "utf-8")
df=pd.read_csv(f3d)     #读入数据
data_3d=df.iloc[:7,1:50]
data3d = df.iloc[start1:end1,1:50]

f3q=open(r'C:\Users\310\Desktop\MTE-LSTM\1213high schools\Primary Schools\Creigiau Primary School\places.857_2013_gas.csv',encoding= "utf-8")
df=pd.read_csv(f3q)     #读入数据
data_3q=df.iloc[:7,1:50]
data3q = df.iloc[start1:end1,1:50]

data3d = data3d.T
data3q = data3q.T
df3dd = data3d
df3d=pd.concat(df3dd.iloc[:,i] for i in range(df3dd.shape[1]))
df3d.index=np.arange(len(df3d))
df3qq = data3q
df3q=pd.concat(df3qq.iloc[:,i] for i in range(df3qq.shape[1]))
df3q.index=np.arange(len(df3q))
data3 = pd.DataFrame([df3q,df3d])
data3 = data3.T
data3 = np.array(data3)
data3.shape
data3


f4d=open(r'C:\Users\310\Desktop\MTE-LSTM\1213high schools\Primary Schools\Glyncoed Primary School\places.860_2013_elec.csv',encoding= "utf-8")
df=pd.read_csv(f4d)     #读入数据
data_4d=df.iloc[:7,1:50]
data4d = df.iloc[start1:end1,1:50]

f4q=open(r'C:\Users\310\Desktop\MTE-LSTM\1213high schools\Primary Schools\Glyncoed Primary School\places.860_2013_gas.csv',encoding= "utf-8")
df=pd.read_csv(f4q)     #读入数据
data_4q=df.iloc[:7,1:50]
data4q = df.iloc[start1:end1,1:50]

data4d = data4d.T
data4q = data4q.T
df4dd = data4d
df4d=pd.concat(df4dd.iloc[:,i] for i in range(df4dd.shape[1]))
df4d.index=np.arange(len(df4d))
df4qq = data4q
df4q=pd.concat(df4qq.iloc[:,i] for i in range(df4qq.shape[1]))
df4q.index=np.arange(len(df4q))
data4 = pd.DataFrame([df4q,df4d])
data4 = data4.T
data4 = np.array(data4)




f5d=open(r'C:\Users\310\Desktop\MTE-LSTM\1213high schools\Primary Schools\Christ The King R.C Primary School\places.995_2013_elec.csv',encoding= "utf-8")
df=pd.read_csv(f5d)     #读入数据
data_5d=df.iloc[:7,1:50]
data5d = df.iloc[start1:end1,1:50]

f5q=open(r'C:\Users\310\Desktop\MTE-LSTM\1213high schools\Primary Schools\Christ The King R.C Primary School\places.995_2013_gas.csv',encoding= "utf-8")
df=pd.read_csv(f5q)     #读入数据
data_5q=df.iloc[:7,1:50]
data5q = df.iloc[start1:end1,1:50]

data5d = data5d.T
data5q = data5q.T
df5dd = data5d
df5d=pd.concat(df5dd.iloc[:,i] for i in range(df5dd.shape[1]))
df5d.index=np.arange(len(df5d))
df5qq = data5q
df5q=pd.concat(df5qq.iloc[:,i] for i in range(df5qq.shape[1]))
df5q.index=np.arange(len(df5q))
data5 = pd.DataFrame([df5q,df5d])
data5 = data5.T
data5 = np.array(data5)

f6d=open(r'C:\Users\310\Desktop\MTE-LSTM\1213high schools\Primary Schools\Gwaelod Y Garth Primary School\places.959_2013_elec.csv',encoding= "utf-8")
df=pd.read_csv(f6d)     #读入数据
data_6d=df.iloc[:7,1:50]
data6d = df.iloc[start1:end1,1:50]

f6q=open(r'C:\Users\310\Desktop\MTE-LSTM\1213high schools\Primary Schools\Gwaelod Y Garth Primary School\places.959_2013_gas.csv',encoding= "utf-8")
df=pd.read_csv(f6q)     #读入数据
data_6q=df.iloc[:7,1:50]
data6q = df.iloc[start1:end1,1:50]

data6d = data6d.T
data6q = data6q.T
df6dd = data6d
df6d=pd.concat(df6dd.iloc[:,i] for i in range(df6dd.shape[1]))
df6d.index=np.arange(len(df6d))
df6qq = data6q
df6q=pd.concat(df6qq.iloc[:,i] for i in range(df6qq.shape[1]))
df6q.index=np.arange(len(df6q))
data6 = pd.DataFrame([df6q,df6d])
data6 = data6.T
data6 = np.array(data6)


f7d=open(r'C:\Users\310\Desktop\MTE-LSTM\1213high schools\Primary Schools\Herbert Thompson Primary School\places.923_2013_elec.csv',encoding= "utf-8")
df=pd.read_csv(f7d)     #读入数据
data_7d=df.iloc[:7,1:50]
data7d = df.iloc[start1:end1,1:50]

f7q=open(r'C:\Users\310\Desktop\MTE-LSTM\1213high schools\Primary Schools\Herbert Thompson Primary School\places.923_2013_gas.csv',encoding= "utf-8")
df=pd.read_csv(f7q)     #读入数据
data_7q=df.iloc[:7,1:50]
data7q = df.iloc[start1:end1,1:50]

data7d = data7d.T
data7q = data7q.T
df7dd = data7d
df7d=pd.concat(df7dd.iloc[:,i] for i in range(df7dd.shape[1]))
df7d.index=np.arange(len(df7d))
df7qq = data7q
df7q=pd.concat(df7qq.iloc[:,i] for i in range(df7qq.shape[1]))
df7q.index=np.arange(len(df7q))
data7 = pd.DataFrame([df7q,df7d])
data7 = data7.T
data7 = np.array(data7)


f8d=open(r'C:\Users\310\Desktop\MTE-LSTM\1213high schools\Primary Schools\Kitchener Primary School\places.927_2013_elec.csv',encoding= "utf-8")
df=pd.read_csv(f8d)     #读入数据
data_8d=df.iloc[:7,1:50]
data8d = df.iloc[start1:end1,1:50]

f8q=open(r'C:\Users\310\Desktop\MTE-LSTM\1213high schools\Primary Schools\Kitchener Primary School\places.927_2013_gas.csv',encoding= "utf-8")
df=pd.read_csv(f8q)     #读入数据
data_8q=df.iloc[:7,1:50]
data8q = df.iloc[start1:end1,1:50]

data8d = data8d.T
data8q = data8q.T
df8dd = data8d
df8d=pd.concat(df8dd.iloc[:,i] for i in range(df8dd.shape[1]))
df8d.index=np.arange(len(df8d))
df8qq = data8q
df8q=pd.concat(df8qq.iloc[:,i] for i in range(df8qq.shape[1]))
df8q.index=np.arange(len(df8q))

data8 = pd.DataFrame([df8q,df8d])
data8 = data8.T
data8 = np.array(data8)



f9d=open(r'C:\Users\310\Desktop\MTE-LSTM\1213high schools\Primary Schools\Roath Park Primary School\places.962_2013_elec.csv',encoding= "utf-8")
df=pd.read_csv(f9d)     #读入数据
data_9d=df.iloc[:7,1:50]
data9d = df.iloc[start1:end1,1:50]

f9q=open(r'C:\Users\310\Desktop\MTE-LSTM\1213high schools\Primary Schools\Roath Park Primary School\places.962_2013_gas.csv',encoding= "utf-8")
df=pd.read_csv(f9q)     #读入数据
data_9q=df.iloc[:7,1:50]
data9q = df.iloc[start1:end1,1:50]

data9d = data9d.T
data9q = data9q.T
df9dd = data9d
df9d=pd.concat(df9dd.iloc[:,i] for i in range(df9dd.shape[1]))
df9d.index=np.arange(len(df9d))
df9qq = data9q
df9q=pd.concat(df9qq.iloc[:,i] for i in range(df9qq.shape[1]))
df9q.index=np.arange(len(df9q))
data9 = pd.DataFrame([df9q,df9d])
data9 = data9.T
data9 = np.array(data9)


f10d=open(r'C:\Users\310\Desktop\MTE-LSTM\1213high schools\Primary Schools\Lakeside Primary School\places.933_2013_elec.csv',encoding= "utf-8")
df=pd.read_csv(f10d)     #读入数据
data_10d=df.iloc[:7,1:50]
data10d = df.iloc[start1:end1,1:50]

f10q=open(r'C:\Users\310\Desktop\MTE-LSTM\1213high schools\Primary Schools\Lakeside Primary School\places.933_2013_gas.csv',encoding= "utf-8")
df=pd.read_csv(f10q)     #读入数据
data_10q=df.iloc[:7,1:50]
data10q = df.iloc[start1:end1,1:50]

data10d = data10d.T
data10q = data10q.T
df10dd = data10d
df10d=pd.concat(df10dd.iloc[:,i] for i in range(df10dd.shape[1]))
df10d.index=np.arange(len(df10d))
df10qq = data10q
df10q=pd.concat(df10qq.iloc[:,i] for i in range(df10qq.shape[1]))
df10q.index=np.arange(len(df10q))
data10 = pd.DataFrame([df10q,df10d])
data10 = data10.T
data10 = np.array(data10)
data10.shape
data10


f11d=open(r'C:\Users\310\Desktop\MTE-LSTM\1213high schools\Primary Schools\Lansdowne Primary School\places.928_2013_elec.csv',encoding= "utf-8")
df=pd.read_csv(f11d)     #读入数据
data_11d=df.iloc[:7,1:50]
data11d = df.iloc[start1:end1,1:50]

f11q=open(r'C:\Users\310\Desktop\MTE-LSTM\1213high schools\Primary Schools\Lansdowne Primary School\places.928_2013_gas.csv',encoding= "utf-8")
df=pd.read_csv(f11q)     #读入数据
data_11q=df.iloc[:7,1:50]
data11q = df.iloc[start1:end1,1:50]

data11d = data11d.T
data11q = data11q.T
df11dd = data11d
df11d=pd.concat(df11dd.iloc[:,i] for i in range(df11dd.shape[1]))
df11d.index=np.arange(len(df11d))
df11qq = data11q
df11q=pd.concat(df11qq.iloc[:,i] for i in range(df11qq.shape[1]))
df11q.index=np.arange(len(df11q))
data11 = pd.DataFrame([df11q,df11d])
data11 = data11.T
data11 = np.array(data11)



f12d=open(r'C:\Users\310\Desktop\MTE-LSTM\1213high schools\Primary Schools\Llandaff C.W. Primary School\places.922_2013_elec.csv',encoding= "utf-8")
df=pd.read_csv(f12d)     #读入数据
data_12d=df.iloc[:7,1:50]
data12d = df.iloc[start1:end1,1:50]

f12q=open(r'C:\Users\310\Desktop\MTE-LSTM\1213high schools\Primary Schools\Llandaff C.W. Primary School\places.922_2013_gas.csv',encoding= "utf-8")
df=pd.read_csv(f12q)     #读入数据
data_12q=df.iloc[:7,1:50]
data12q = df.iloc[start1:end1,1:50]

data12d = data12d.T
data12q = data12q.T
df12dd = data12d
df12d=pd.concat(df12dd.iloc[:,i] for i in range(df12dd.shape[1]))
df12d.index=np.arange(len(df12d))
df12qq = data12q
df12q=pd.concat(df12qq.iloc[:,i] for i in range(df12qq.shape[1]))
df12q.index=np.arange(len(df12q))
data12 = pd.DataFrame([df12q,df12d])
data12 = data12.T
data12 = np.array(data12)



f13d=open(r'C:\Users\310\Desktop\MTE-LSTM\1213high schools\Primary Schools\Llanedeyrn Primary School\places.1030_2013_elec.csv',encoding= "utf-8")
df=pd.read_csv(f13d)     #读入数据
data_13d=df.iloc[:7,1:50]
data13d = df.iloc[start1:end1,1:50]

f13q=open(r'C:\Users\310\Desktop\MTE-LSTM\1213high schools\Primary Schools\Llanedeyrn Primary School\places.1030_2013_gas.csv',encoding= "utf-8")
df=pd.read_csv(f13q)     #读入数据
data_13q=df.iloc[:7,1:50]
data13q = df.iloc[start1:end1,1:50]

data13d = data13d.T
data13q = data13q.T
df13dd = data13d
df13d=pd.concat(df13dd.iloc[:,i] for i in range(df13dd.shape[1]))
df13d.index=np.arange(len(df13d))
df13qq = data13q
df13q=pd.concat(df13qq.iloc[:,i] for i in range(df13qq.shape[1]))
df13q.index=np.arange(len(df13q))
data13 = pd.DataFrame([df13q,df13d])
data13 = data13.T
data13 = np.array(data13)
data13.shape
data13

f14d=open(r'C:\Users\310\Desktop\MTE-LSTM\1213high schools\Primary Schools\Moorland Primary School\places.862_2013_elec.csv',encoding= "utf-8")
df=pd.read_csv(f14d)     #读入数据
data_14d=df.iloc[:7,1:50]
data14d = df.iloc[start1:end1,1:50]

f14q=open(r'C:\Users\310\Desktop\MTE-LSTM\1213high schools\Primary Schools\Moorland Primary School\places.862_2013_gas.csv',encoding= "utf-8")
df=pd.read_csv(f14q)     #读入数据
data_14q=df.iloc[:7,1:50]
data14q = df.iloc[start1:end1,1:50]

data14d = data14d.T
data14q = data14q.T
df14dd = data14d

df14d=pd.concat(df14dd.iloc[:,i] for i in range(df14dd.shape[1]))
df14d.index=np.arange(len(df14d))
df14qq = data14q
df14q=pd.concat(df14qq.iloc[:,i] for i in range(df14qq.shape[1]))
df14q.index=np.arange(len(df14q))
data14 = pd.DataFrame([df14q,df14d])
data14 = data14.T
data14 = np.array(data14)
data14.shape
data14


f15d=open(r'C:\Users\310\Desktop\MTE-LSTM\1213high schools\Primary Schools\Mount Stuart Primary School\places.817_2013_elec.csv',encoding= "utf-8")
df=pd.read_csv(f15d)     #读入数据
data_15d=df.iloc[:7,1:50]
data15d = df.iloc[start1:end1,1:50]

f15q=open(r'C:\Users\310\Desktop\MTE-LSTM\1213high schools\Primary Schools\Mount Stuart Primary School\places.817_2013_gas.csv',encoding= "utf-8")
df=pd.read_csv(f15q)     #读入数据
data_15q=df.iloc[:7,1:50]
data15q = df.iloc[start1:end1,1:50]

data15d = data15d.T
data15q = data15q.T
df15dd = data15d
df15d=pd.concat(df15dd.iloc[:,i] for i in range(df15dd.shape[1]))
df15d.index=np.arange(len(df15d))
df15qq = data15q
df15q=pd.concat(df15qq.iloc[:,i] for i in range(df15qq.shape[1]))
df15q.index=np.arange(len(df15q))
data15 = pd.DataFrame([df15q,df15d])
data15 = data15.T
data15 = np.array(data15)


f16d=open(r'C:\Users\310\Desktop\MTE-LSTM\1213high schools\Primary Schools\Ninian Park Primary School\places.996_2013_elec.csv',encoding= "utf-8")
df=pd.read_csv(f16d)     #读入数据
data_16d=df.iloc[:7,1:50]
data16d = df.iloc[start1:end1,1:50]

f16q=open(r'C:\Users\310\Desktop\MTE-LSTM\1213high schools\Primary Schools\Ninian Park Primary School\places.996_2013_gas.csv',encoding= "utf-8")
df=pd.read_csv(f16q)     #读入数据
data_16q=df.iloc[:7,1:50]
data16q = df.iloc[start1:end1,1:50]

data16d = data16d.T
data16q = data16q.T
df16dd = data16d
df16d=pd.concat(df16dd.iloc[:,i] for i in range(df16dd.shape[1]))
df16d.index=np.arange(len(df16d))
df16qq = data16q
df16q=pd.concat(df16qq.iloc[:,i] for i in range(df16qq.shape[1]))
df16q.index=np.arange(len(df16q))
data16 = pd.DataFrame([df16q,df16d])
data16 = data16.T
data16 = np.array(data16)



f17d=open(r'C:\Users\310\Desktop\MTE-LSTM\1213high schools\Primary Schools\Peter Lea Primary School\places.971_2013_elec.csv',encoding= "utf-8")
df=pd.read_csv(f17d)     #读入数据
data_17d=df.iloc[:7,1:50]
data17d = df.iloc[start1:end1,1:50]

f17q=open(r'C:\Users\310\Desktop\MTE-LSTM\1213high schools\Primary Schools\Peter Lea Primary School\places.971_2013_gas.csv',encoding= "utf-8")
df=pd.read_csv(f17q)     #读入数据
data_17q=df.iloc[:7,1:50]
data17q = df.iloc[start1:end1,1:50]

data17d = data17d.T
data17q = data17q.T
df17dd = data17d
df17d=pd.concat(df17dd.iloc[:,i] for i in range(df17dd.shape[1]))
df17d.index=np.arange(len(df17d))
df17qq = data17q
df17q=pd.concat(df17qq.iloc[:,i] for i in range(df17qq.shape[1]))
df17q.index=np.arange(len(df17q))
data17 = pd.DataFrame([df17q,df17d])
data17 = data17.T
data17 = np.array(data17)


f18d=open(r'C:\Users\310\Desktop\MTE-LSTM\1213high schools\Primary Schools\Rhiwbina Primary School\places.979_2013_elec.csv',encoding= "utf-8")
df=pd.read_csv(f18d)     #读入数据
data_18d=df.iloc[:7,1:50]
data18d = df.iloc[start1:end1,1:50]

f18q=open(r'C:\Users\310\Desktop\MTE-LSTM\1213high schools\Primary Schools\Rhiwbina Primary School\places.979_2013_gas.csv',encoding= "utf-8")
df=pd.read_csv(f18q)     #读入数据
data_18q=df.iloc[:7,1:50]
data18q = df.iloc[start1:end1,1:50]

data18d = data18d.T
data18q = data18q.T
df18dd = data18d
df18d=pd.concat(df18dd.iloc[:,i] for i in range(df18dd.shape[1]))
df18d.index=np.arange(len(df18d))
df18qq = data18q
df18q=pd.concat(df18qq.iloc[:,i] for i in range(df18qq.shape[1]))
df18q.index=np.arange(len(df18q))
data18 = pd.DataFrame([df18q,df18d])
data18 = data18.T
data18 = np.array(data18)





def dtw_distance(ts_a, ts_b, d=lambda x,y: abs(x-y), mww=10000):
    """
    Computes dtw distance between two time series
    
    Args:
        ts_a: time series a
        ts_b: time series b
        d: distance function
        mww: max warping window, int, optional (default = infinity)
        
    Returns:
        dtw distance
    """
    
    # Create cost matrix via broadcasting with large int
    ts_a, ts_b = np.array(ts_a), np.array(ts_b)
    M, N = len(ts_a), len(ts_b)
    cost = np.ones((M, N))

    # Initialize the first row and column
    cost[0, 0] = d(ts_a[0], ts_b[0])
    for i in range(1, M):
        cost[i, 0] = cost[i-1, 0] + d(ts_a[i], ts_b[0])

    for j in range(1, N):
        cost[0, j] = cost[0, j-1] + d(ts_a[0], ts_b[j])

    # Populate rest of cost matrix within window
    for i in range(1, M):
        for j in range(max(1, i - mww), min(N, i + mww)):
            choices = cost[i-1, j-1], cost[i, j-1], cost[i-1, j]
            cost[i, j] = min(choices) + d(ts_a[i], ts_b[j])

    # Return DTW distance given window 
    return cost[-1, -1]


d1 = pd.DataFrame([df1d,df1q])
d1 = d1.T
d1 =np.array(d1)


d2 = pd.DataFrame([df2d,df2q])
d2 = d2.T
d2 =np.array(d2)


d3 = pd.DataFrame([df3d,df3q])
d3 = d3.T
d3 =np.array(d3)


d4 = pd.DataFrame([df4d,df4q])
d4 = d4.T
d4 =np.array(d4)


d5 = pd.DataFrame([df5d,df5q])
d5 = d5.T
d5 =np.array(d5)


d6 = pd.DataFrame([df6d,df6q])
d6 = d6.T
d6 =np.array(d6)


d7 = pd.DataFrame([df7d,df7q])
d7 = d7.T
d7 =np.array(d7)


d8 = pd.DataFrame([df8d,df8q])
d8 = d8.T
d8 =np.array(d8)


#d18 = pd.DataFrame([df18d,df18q])
d9 = pd.DataFrame([df9d,df9q])
d9 = d9.T
d9 =np.array(d9)


d10 = pd.DataFrame([df10d,df10q])
d10 = d10.T
d10 =np.array(d10)


d11 = pd.DataFrame([df11d,df11q])
d11 = d11.T
d11 =np.array(d11)


d12 = pd.DataFrame([df12d,df12q])
d12 = d12.T
d12 =np.array(d12)


d13 = pd.DataFrame([df13d,df13q])
d13 = d13.T
d13 =np.array(d13)


d14 = pd.DataFrame([df14d,df14q])
d14 = d14.T
d14 =np.array(d14)


d15 = pd.DataFrame([df15d,df15q])
d15 = d15.T
d15 =np.array(d15)


d16 = pd.DataFrame([df16d,df16q])
d16 = d16.T
d16 =np.array(d16)


d17 = pd.DataFrame([df17d,df17q])
d17 = d17.T
d17 =np.array(d17)


d18 = pd.DataFrame([df18d,df18q])
d18 = d16.T
d18 =np.array(d18)


dtw = dtw_distance(df1d,df14d)
print(dtw)
