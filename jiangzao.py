import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM
from keras.layers import Dense , Dropout
from keras.models import Sequential
from sklearn.metrics import mean_squared_error
import numpy as np
import math
from sklearn import metrics
import torch
from vmdpy import VMD
from numpy import array
import pywt
data = pd.read_csv(r"TBMdata1.csv")
print(data.head())
print(f"len(data):{len(data)}")
# all_data = data.loc[:,['Cutter_speed','Cutter_Torque','Penetration','thrust','advance_speed_percent']].values
all_data = data.values
Penetration = data['Penetration'].values
print(f"len(Penetration):{len(Penetration)}")
print(Penetration)
print(all_data)

d = [210893,8]
threshold = 0.2
wavelet = 'db2'
level = 1
# data = array(data).flatten()  # 转化为一维数组，便于降噪
for j in range(0,8,1):
  data=all_data[:,j]
  print(data)
  data = data.transpose()
  data = pywt.wavedec(data, wavelet, level=level)  # 进行小波分解
  # 对细节系数进行阈值处理
  data[1:] = [pywt.threshold(i, threshold * max(i), mode='soft') for i in data[1:]]
  data = pywt.waverec(data, wavelet)
  print(data.shape)
  data = np.array(data).reshape(-1, 210894)
  print(data.shape)
  print(data)
  for p in range(0,210893):
      all_data[p,j] = data[0,p]


df = pd.DataFrame(all_data)
df.to_csv("TBMjiangzao1.csv")