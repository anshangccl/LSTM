import pandas as pd
import matplotlib.pyplot as plt
import tf
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM
from keras.layers import Dense , Dropout
from keras.models import Sequential
import keras
from sklearn.metrics import mean_squared_error
import numpy as np
import math
from sklearn import metrics
import torch
from vmdpy import VMD
from keras.models import load_model
from numpy import array
import pywt
# import register_keras_serializable
import pickle
##忽略提醒
import warnings
warnings.filterwarnings("ignore")
#读入数据，简单查看一下数据的前几行
data = pd.read_csv(r"TBMguiyihua.csv")
print(data.head())
print(f"len(data):{len(data)}")
# all_data = data.loc[:,['Cutter_speed','Cutter_Torque','Penetration','thrust','advance_speed_percent']].values
all_data = data.values
Penetration = data['Penetration'].values
print(f"len(Penetration):{len(Penetration)}")
print(Penetration)
print(all_data)
# Penetration = Penetration.transpose()
# Penetration = array(Penetration).flatten()
# # 创建MinMaxScaler对象
# scaler = MinMaxScaler()
# # 将数据进行归一化
# for j in range(0,8,1):
#     data = all_data[:,j]
#     print(data)
#     data = scaler.fit_transform(data.reshape(-1,1))
#     print(data)
#     data = np.array(data).reshape(-1, 210893)
#     for p in range(0, 210893):
#         all_data[p, j] = data[0,p]
# Penetration = scaler.fit_transform(Penetration.reshape(-1,1))
# df = pd.DataFrame(all_data)
# df.to_csv("TBMguiyihua.csv")
data1 = pd.read_csv(r"TBMjiangzao1.csv")
Penetration1 = data1['Penetration'].values
print(Penetration1)
scaler = MinMaxScaler()
Penetration1 = scaler.fit_transform(Penetration1.reshape(-1,1))
print(Penetration1)


def split_data1(data1,time_step=10):#划分数据，以前10组数据。预测之后的一个数据
    data_y=[]
    for i in range(len(data1)-time_step):
        data_y.append(data1[i+time_step])
    data_y=np.array(data_y)
    return data_y
data_y=split_data1(Penetration,time_step=10)
print(f"datay.shape:{data_y.shape}")
def split_data2(data2,time_step):
    data_X = []
    for i in range(len(data2)-time_step):
        data_X.append(data2[i:i+time_step])
    data_X = np.array(data_X)
    return data_X
#划分训练集和测试集的函数
data_X = split_data2(all_data,time_step=10)
print(f"dataX.shape:{data_X.shape}")

def train_test_split(datay, dataX,shuffle=False,percentage=0.8):
    """
    将训练数据X和标签y以numpy.array数组的形式传入
    划分的比例定为训练集:测试集=8:2
    """
    if shuffle:
        random_num=[index for index in range(len(dataX))]
        np.random.shuffle(random_num)
        dataX = dataX[random_num]
        datay = datay[random_num]
    #     shuffle 会将顺序打乱，数据就不具有时序性
    # split_num1 = int(len(data_y)*percentage)#训练集个数
    # print(split_num1)
    split_num1 = int(len(datay) * percentage)
    train_y = datay[:split_num1]
    test_y = datay[split_num1:]
    train_X = dataX[:split_num1, :,:]
    test_X = dataX[split_num1:, :,:]

    return train_y, test_y, train_X, test_X
train_y, test_y, train_X, test_X  = train_test_split(data_y,data_X, shuffle = False,percentage = 0.8)
print(f"train_y.shape:{train_y.shape},test_y.shape:{test_y.shape}")
print(f"train_X.shape:{train_X.shape},test_X.shape:{test_X.shape}")
X_train,y_train = train_X,train_y



#模型建立
model = Sequential()
# set the first hidden layer and set the input dimension
model.add(LSTM(
    input_shape=(30, 8), units = 32, return_sequences=True
))
model.add(Dropout(0.4))

# add the second layer
model.add(LSTM(
    units = 64, return_sequences = False
))
model.add(Dropout(0.4))

# add the output layer with a Dense
model.add(Dense(units=1, activation='linear'))
model.compile(loss='mse', optimizer='adam')
# train the model and use the validation part to validateD
history = model.fit(train_X, train_y, batch_size = 2000, epochs = 200, validation_split = 0.2)
# # 将 History 对象转换为字典
# history_dict = history.history
#
# # 使用 pickle 保存 History 对象
# with open('history.pkl', 'wb') as file_pi:
#     pickle.dump(history_dict, file_pi)
# model.save('moxing.hdf5')


# do the prediction
y_predicted = model.predict(test_X)
test_X2 = torch.Tensor(test_X)
test_y1 = torch.Tensor(test_y)
train_X2 = torch.Tensor(train_X)
train_y1 = torch.Tensor(train_y)
# train_pred=model().numpy()
# test_pred=model(test_X2).numpy()
# pred_y=np.concatenate((train_pred,test_pred))
pred_y=scaler.inverse_transform(y_predicted.reshape(-1,1)).T[0]
# true_y=np.concatenate((test_y))
true_y=scaler.inverse_transform(test_y.reshape(-1,1)).T[0]
# r2_1=metrics.r2_score(train_pred, y_train)
r2_2=metrics.r2_score(y_predicted, test_y)
# 提取训练损失和验证损失
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# 绘制训练损失和验证损 失
epochs = range(1, len(train_loss) + 1)
plt.figure(figsize=(12, 6))
plt.plot(epochs, train_loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# print('训练集决定系数：',r2_1)
print('测试集决定系数：',r2_2)
plt.figure(figsize=(14, 6))
plt.plot(pred_y,marker="o",markersize=1,label="pred_y")
plt.plot(true_y,marker="x",markersize=1,label="true_y")
plt.title("LSTM")
# plt.plot(epochs , loss, marker="-",markersize=1,label="mse")
plt.legend()
plt.show()