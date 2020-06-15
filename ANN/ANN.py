import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#Reading Data
dataset = pd.read_csv('https://raw.githubusercontent.com/krishnaik06/Complete-Deep-Learning/master/ANN/Churn_Modelling.csv')
x=dataset.iloc[:,3:13]
y=dataset.iloc[:,13]
#Getting the catagorical feature out
geography=pd.get_dummies(x['Geography'],drop_first=True)
gender=pd.get_dummies(x['Gender'],drop_first=True)
x=x.drop(['Geography','Gender'],axis=1)
x=pd.concat([x,geography,gender],axis=1)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import LeakyReLU,PReLU,ELU
from keras.layers import Dense
from keras.layers import Dropout

classifier=Sequential()

classifier.add(Dense(units=6,kernel_initializer='he_uniform',activation='relu',input_dim=11))

classifier.add(Dense(units=6,kernel_initializer='he_uniform',activation='relu'))

classifier.add(Dense(units=1,kernel_initializer='glorot_uniform',activation='sigmoid'))

classifier.compile(optimizer='Adamax',loss='binary_crossentropy',metrics=['accuracy'])

model=classifier.fit(X_train,y_train,validation_split=0.33,batch_size=10,nb_epoch=100)

