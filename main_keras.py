#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense
#from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.utils import to_categorical
#from sklearn.model_selection import train_test_split
#import numpy as np
#from facilities.StaticFuncs import GetImgAndOneHotEncodedLabel
#
## 載入資料集
#sDatasetDir = './ChineseMNIST'
#
## 讀資料
#X_train, y_train, X_test, y_test, dictLabelCat = GetImgAndOneHotEncodedLabel(sDatasetDir)
## 每個樣本 reshape 成一維並根據要求作正規化(/255)
#X_train, X_test = tuple(map(lambda x: funcReshapeAndNomalization(x), [X_train, X_test]))
#
##data = cp.load('cmnist_data.npz')
##X = data['x_train']
##y = data['y_train']
##
### 將 y 轉成 one-hot 向量
##y = to_categorical(y)
##
### 切分訓練集與測試集
##X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
## 建立 FNN 模型
#model = Sequential()
#model.add(Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
#model.add(Dense(128, activation='relu'))
#model.add(Dense(64, activation='relu'))
#model.add(Dense(15, activation='softmax'))
#
## 編譯模型
#optimizer = Adam(lr=0.001)
#model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
#
## 訓練模型
#model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))
#
## 評估模型
#loss, accuracy = model.evaluate(X_test, y_test)
#print('Test accuracy:', accuracy)

import cupy as cp
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adam, SGD
from keras.utils import np_utils

from facilities.StaticFuncs import GetImgAndOneHotEncodedLabel, funcReshapeAndNomalization

# 載入資料集
sDatasetDir = './ChineseMNIST'

# 讀資料
X_train, y_train, X_test, y_test, dictLabelCat = GetImgAndOneHotEncodedLabel(sDatasetDir)
# 每個樣本 reshape 成一維並根據要求作正規化(/255)
X_train, X_test = tuple(map(lambda x: funcReshapeAndNomalization(x), [X_train, X_test]))

X_train, X_test = np.array(X_train.get()), np.array(X_test.get())
y_train, y_test = np.array(y_train.get()), np.array(y_test.get())

# 建立 FNN 模型
model = Sequential()
model.add(Dense(units=2048, activation='relu', input_dim=4096))
model.add(Dropout(0.3))
model.add(Dense(units=1024, activation='relu', input_dim=2048))
model.add(Dropout(0.2))
model.add(Dense(units=512, activation='relu', input_dim=1024))
model.add(Dropout(0.2))
model.add(Dense(units=256, activation='relu', input_dim=512))
model.add(Dropout(0.2))
model.add(Dense(units=128, activation='LeakyReLU', input_dim=256))
model.add(Dropout(0.2))
model.add(Dense(units=64, activation='LeakyReLU', input_dim=128))
model.add(Dropout(0.1))
model.add(Dense(units=32, activation='LeakyReLU', input_dim=64))
model.add(Dropout(0.1))
model.add(Dense(units=15, activation='softmax'))

# 定義優化器、損失函數、評估指標
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# 訓練模型
model.fit(X_train, y_train, batch_size=128, epochs=80, validation_data=(X_test, y_test))
