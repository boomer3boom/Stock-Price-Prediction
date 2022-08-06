#Import library
from ast import increment_lineno
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20, 10

from tensorflow.keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense

from sklearn.preprocessing import MinMaxScaler

# %%
df = pd.read_csv("NSE-TATA.csv")
df.head()

df['Date'] = pd.to_datetime(df.Date, format = '%Y-%m-%d')