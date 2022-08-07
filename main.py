#Import library
from ast import increment_lineno
from dataclasses import dataclass
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
    new_dataset["Date"][i]=data['Date'][i]
    new_dataset["Close"][i]=data["Close"][i]


# %%
"""
Normalise the training data so that they all have the
same format
"""

#scaler can transform all given input into (0, 1)
scaler=MinMaxScaler(feature_range=(0,1))

#return the values of new_dataset in nested list
final_dataset=new_dataset.values

print(final_dataset)