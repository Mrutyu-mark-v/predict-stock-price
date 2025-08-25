import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler


model = load_model('/home/hell/Project/predict-stock-price/stock-prediction-model.keras')

st.header('Stock market prediction')

stock = st.text_input('Enter stock symbol','GOOG')
start = '2012-01-01'
end = '2024-12-01'

data = yf.download(stock,start,end)

st.subheader('stock data')
st.write(data)

data_train = pd.DataFrame(data.Close[0: int(len(data)*0.8)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

scaler = MinMaxScaler(feature_range=(0,1))

past_100_days = data_train.tail(100)
data_test = pd.concat([past_100_days,data_train], ignore_index=True)

data_test_scaler = scaler.fit_transform(data_test)

x=[]
y=[]
for i in range(100, data_test_scaler.shape[0]):
    x.append(data_test_scaler[i-100:i])
    y.append(data_test_scaler[i,0])

x,y = np.array(x),np.array(y)