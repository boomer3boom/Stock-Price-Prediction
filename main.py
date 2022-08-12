#Import library
from ast import increment_lineno
from dataclasses import dataclass
from msilib import sequence
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20, 10

from tensorflow.keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense

from sklearn.preprocessing import MinMaxScaler

# %%
"""
The goal in this section is to open Tata Global Beverages Limited stock 
value.
"""
df = pd.read_csv("NSE-TATA.csv")
df.head()


#%%
"""
The goal in this section is to create a graph that let us analyse the
closing prices.
"""

#The current date column looks like it's in pd datetime, but it's not.
df["Date"]=pd.to_datetime(df.Date,format="%Y-%m-%d")
df.index=df['Date']

#Create the figure for analysis with a width of 16, height 8.
plt.figure(figsize=(16,8))
#plot our graph with y axis as the close value, and x axis as index (date)
plt.plot(df["Close"],label='Close Price history')


#%%
"""
The goal here is to sort our dataset with the datetime and set them into a 
new dataset. This dataset only has the date and closing value of the stock
"""
#sort our dataframe along each rows (Datetime) and assign that to data
data = df.sort_index(ascending = True, axis = 0)

#Create na new dataframe With Date and Close (closing value) as columns
new_dataset=pd.DataFrame(index=range(0,len(df)),columns=['Date','Close'])

#Convert our Date and Close in Daa into new_dataset
for i in range(0,len(data)):
    #Assign the ith position of Date/Close to the ith position in data
    new_dataset["Date"][i]=data["Date"][i]
    new_dataset["Close"][i]=data["Close"][i]


# %%
"""
Normalise the training data so that they all have the
same format. Also filter these data into training,
validation, as well as it's x and y component
"""
#scaler=MinMaxScaler(feature_range=(0,1))

#return the values of new_dataset in nested list
final_dataset=new_dataset.values

#The training and validation component of dataset
train_data=final_dataset[0:987,:]
valid_data=final_dataset[987:,:]

#scaler can transform all given input into (0, 1)
scaler=MinMaxScaler(feature_range=(0,1))

#Transform our dataset into the range of (0,1)
scaled_data=scaler.fit_transform(final_dataset)

x_train_data,y_train_data=[],[]

#Append our scaled data into 
for i in range(60,len(train_data)):

    #We need to cover 60-timestamp for the RNN to predict the 61st price. 
    #So x_train is nested each containing 60-time stamp prices
    x_train_data.append(scaled_data[i-60:i,0])

    #y_train contains the stock price everyday which is the stock price 
    #corresponding to reach list in x_train.
    y_train_data.append(scaled_data[i,0])

x_train_data,y_train_data=np.array(x_train_data),np.array(y_train_data)

#reshape into (x_train_data.shape[0],x_train_data.shape[1], 1) -> (927,60,1)
#This is in the form (batch size, number of timestamp, number of key features)
x_train_data=np.reshape(x_train_data,
    (x_train_data.shape[0],x_train_data.shape[1],1))

# %%
"""
Building the long short term memmory model for our model to memorise 
things and apply it to prediction.
"""

#The sequential model allows a linear stack of layers
lstm_model = Sequential(
    [
    #The output is 50 unit, and return the full sequence. 
    LSTM(units=50,return_sequences=True, input_shape = (x_train_data.shape[1],2)),
    LSTM(units=50),
    #Dense makes it so all neruron recieves input from neurons in the previous layer
    #Unit is 1, since we only have 1 prediction.
    Dense(1)
    ]
)

lstm_model.summary()

#%%
"""
Compiling our model and training it
"""

lstm_model.compile(loss='mean_squared_error',optimizer='adam')
lstm_model.fit(x_train_data,y_train_data,epochs=1,batch_size=1,verbose=2)