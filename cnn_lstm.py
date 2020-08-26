import pandas as pd
from pandas.tseries.offsets import BDay
import datetime
from holidays_jp import CountryHolidays

from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
import keras
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras import optimizers

#Basic functions for parsing the initial data loaded from an Excel file
def bsd(a):
    isBusinessDay = BDay().is_on_offset #instead of onOffset
    match_series = pd.to_datetime(a.index).map(isBusinessDay) #bC is the DataFrame we work on
    a=a[match_series]
    a=a.astype(float)
    return a

def proc(t): #t here is the name of the excel file
    fi=pd.read_excel(t, index_col=0) #skiprows=3 allows you to start at 3/4
    g=fi.drop(fi.index[0])
    g=bsd(g)
    return g

#LOADING the data
"""
This is the file holding the market data. Unfortunately, I cannot make the data set public. 
The structure of the excel file is prettu basic: column A holds the dates, and then each column after B holds a stock
"""

t='path_to_market_data_file_name.xlsx' #use YOUR own file here
bb=proc(t)

#GETTING RID of the JAPANESE holidays from the data. You can use, add or subtract the relevant years for your data in the list below

years=[2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
hol=[]
for k in years:
    holidays = CountryHolidays.get('JP', k)
    for j in holidays:
        hol.append(j[0])

def elim_holidays(bb, hol):
    for k in range(len(bb)-1, 0, -1):
        if bb.index[k] in hol:    #to py_datetime() when we have a Timestamp
            bb=bb.drop(bb.index[k])
    return bb

bb=elim_holidays(bb, hol)
bb=bb[-400:]

#BUILDING the INPUTS and OUTPUT variable to use with the network. The goal is to identify stocks that can jump 10% over 5 days.
def mmxx2(bb):
    x_values=[]
    y_values=[]
    li=list(bb)
    for j in li:
        #preparing the individual stocks
        ss=bb[j]
        #getting rid of the zeros in stock prices
        ss=ss.dropna()
        ss=ss.pct_change()
        ss=ss.dropna()
        ss=ss*100
        i=34
        while i<len(ss)-7:
            s10=ss[i:i+5].sum()
            prev2=ss[i-32:i]
            var_list=list(prev2.values)
            x_values.append(var_list)
            y_values.append(s10)
            i=i+1
    return array(x_values), array(y_values)

X, y=mmxx2(bb) #building the inputs and output variables

n_features = 1
n_seq = 6 #the number of subsequences
n_steps = 5 #number of steps per subsequence
X = X.reshape((X.shape[0], n_seq, n_steps, n_features))

#BUILDING the MODEL to TRAIN - this is the model I used eventually, but you'll need a GPU to get resutls faster
model = Sequential()
model.add(TimeDistributed(Conv1D(filters=256, kernel_size=4, activation='relu'), input_shape=(None, n_steps, n_features)))
model.add(TimeDistributed(Conv1D(filters=512, kernel_size=4, activation='relu', padding='same')))
model.add(TimeDistributed(Conv1D(filters=1024, kernel_size=4, activation='relu', padding='same')))
model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(1024, activation='relu', return_sequences=True))
model.add(LSTM(64))
model.add(Dense(1))
adam = optimizers.Adam(lr=0.0001)
model.compile(optimizer=adam, loss='mse')

#TRAINING the model
model.fit(X, y, epochs=200)

#PREDICTING the values. Here ll is a single stock panda series, ll=bb[STOCK]
def pred(ll):
    ll=ll[-30:]
    ll=list(ll.values)
    x_input = array(ll)
    x_input = x_input.reshape((1, n_seq, n_steps, n_features))
    yhat = model.predict(x_input)
    return yhat
    
