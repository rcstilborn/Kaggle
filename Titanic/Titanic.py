
# coding: utf-8

# # Kaggle Titanic
# 
# ### Data preparation
# 
# Column | Missing Data | Feature Scaling | Comment
# --- |:---:|:---:|:---
# PassengerId|||Needed for submission
# Survived|||Training label
# Pclass||As is|
# Name|||Ignored
# Sex||Encoded as 0(<) or 1(F)|
# Age|Impute with mean|As is|
# SibSp|Impute with mean|As is|
# Parch|Impute with mean|As is|
# Ticket|||Ignored
# Cabin|||Ignored
# Embarked|||Ignored
# 
# ### Model
#   * 15 hidden layers of 128
#   * Dropout .4
#   * loss='categorical_crossentropy', optimizer='adam'
#   * 500 epochs
# 
# 
# ### Kaggle score
#   * 0.75598
# 
# 
# ### Change history
# Score|Comment
# ---|:---
# 0.75598|First attempt
# 0.73684|Changed Pclass and Sex to one hot encoding
# 0.72248|Undid the last change and rescaled Parch as well
# 0.77990|Removed normalization of Age, Parch, SibSp, Pclass
# 0.75598|Added Fare back in
# 0.76555|Let's go back one step and change the model to only two layers
# 0.78947|Reduced epochs to 125, added validation of 0.2

# In[20]:

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:

# load data
train_data = pd.read_csv(r"./data/train.csv")
test_data = pd.read_csv(r"./data/test.csv")


# In[3]:

train_data.head()


# In[4]:

test_data.head()


# In[5]:

# Delete the data we don't need
cols = ["Name", "Ticket", "Embarked", "Cabin", "Fare"]
train_data.drop(cols, axis=1, inplace=True)
test_data.drop(cols, axis=1, inplace=True)


# In[6]:

# Fill the NaNs in Age, SibSp and Parch with the mean of the training data of that column
def fillna_n(col,n):
    col.fillna(n, inplace=True)
    
mean_age = train_data["Age"].mean()
mean_sibsp = train_data["SibSp"].mean()
mean_parch = train_data["Parch"].mean()

fillna_n(train_data["Age"], mean_age)
fillna_n(test_data["Age"], mean_age)
fillna_n(train_data["SibSp"], mean_sibsp)
fillna_n(test_data["SibSp"], mean_sibsp)
fillna_n(train_data["Parch"], mean_parch)
fillna_n(test_data["Parch"], mean_parch)


# In[7]:

# What NaNs do we still have?
print(train_data.isnull().sum())
print(test_data.isnull().sum())
train_data[train_data.isnull().any(axis=1)]
test_data[test_data.isnull().any(axis=1)]


# In[8]:

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(["male","female"])
train_data["Sex"] = le.transform(train_data["Sex"]) 
test_data["Sex"] = le.transform(test_data["Sex"])
# train_data = pd.get_dummies(train_data,columns=["Sex","Pclass"])
# test_data = pd.get_dummies(test_data,columns=["Sex","Pclass"])
# train_data.head()
# train_data.replace({"Sex":{"male":-1,"female":1}},inplace=True)
# test_data.replace({"Sex":{"male":-1,"female":1}},inplace=True)


# In[9]:

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# for data in [train_data, test_data]:
#     data["Age"] = scaler.fit_transform(data["Age"].values.reshape(-1,1))
#     data["SibSp"] = scaler.fit_transform(data["SibSp"].values.reshape(-1,1))
#     data["Parch"] = scaler.fit_transform(data["Parch"].values.reshape(-1,1))
#     data["Pclass"] = scaler.fit_transform(data["Pclass"].values.reshape(-1,1))

train_data.head()


# In[10]:

train_data.to_csv("./data/train_prepped.csv")
test_data.to_csv("./data/test_prepped.csv")


# In[11]:

#save PassengerId for evaluation and remove from data
test_passenger_id=test_data["PassengerId"]
train_data.drop("PassengerId", axis=1, inplace=True)
test_data.drop("PassengerId", axis=1, inplace=True)

y = pd.get_dummies(train_data['Survived'])
x = train_data.drop("Survived", axis=1)


# In[13]:

# Split the data into train and validation sets 0.8/0.2
# from sklearn.model_selection import train_test_split
# X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)


# In[14]:

from keras.models import Sequential
from keras.optimizers import SGD, RMSprop, Adam
from keras.layers import Dense, Activation, Dropout


# In[23]:

model = Sequential()
model.add(Dense(input_dim=x.shape[1], units=128, kernel_initializer='normal', bias_initializer='zeros'))
model.add(Activation('relu'))

for i in range (0,15):
    model.add(Dense(units=128, kernel_initializer='normal', bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(.4))
    
model.add(Dense(units=2))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[24]:

hist = model.fit(x.values, y.values, epochs=125, verbose=2, validation_split=0.2)


# In[25]:

## TODO: Visualize the training and validation loss of your neural network
print(hist.history.keys())

# summarize history for accuracy
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='lower left')
plt.show()
# summarize history for loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# In[26]:

p_survived = model.predict_classes(test_data.values)


# In[27]:

submission = pd.DataFrame()
submission['PassengerId'] = test_passenger_id
submission['Survived'] = p_survived
submission.to_csv('./data/submission.csv', index=False)


# In[ ]:



