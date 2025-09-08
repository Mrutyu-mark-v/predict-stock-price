import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as mlt

# Load model
model = load_model('/home/hell/Project/predict-stock-price/stock-prediction-model.keras')

st.set_page_config(
    page_title="Stock Market Prediction App",   # Title shown in browser tab
    page_icon="📈",                            # You can use emoji or a .png/.ico path
    layout="wide",                             # "centered" or "wide"
    initial_sidebar_state="expanded"           # "expanded" or "collapsed"
)

st.header('Stock market prediction')

# User input
stock = st.text_input('Enter stock symbol', 'GOOG')
start = '2012-01-01'
end = '2024-12-01'

# Fetch stock data
data = yf.download(stock, start, end)

st.subheader('Stock data')
st.write(data)

# Train/test split
data_train = pd.DataFrame(data.Close[0:int(len(data) * 0.8)])
data_test = pd.DataFrame(data.Close[int(len(data) * 0.8):])

# Scaling
scaler = MinMaxScaler(feature_range=(0, 1))
data_train_scaled = scaler.fit_transform(data_train)

# Use last 100 days from training + test set
past_100_days = data_train.tail(100)
data_test = pd.concat([past_100_days, data_test], ignore_index=True)

# Scale test data (use transform, not fit_transform!)
data_test_scaled = scaler.transform(data_test)

# Moving Average Plot
st.subheader('MOVING-AROUND-50')
ma_50_days = data.Close.rolling(50).mean()
fig1 = mlt.figure(figsize=(10, 8))
mlt.plot(ma_50_days, 'r')
mlt.plot(data.Close, 'g')

st.pyplot(fig1)

# Moving Average Plot
st.subheader('MOVING-AROUND-100')
ma_100_days = data.Close.rolling(100).mean()
fig2 = mlt.figure(figsize=(10, 8))
mlt.plot(ma_50_days, 'r')
mlt.plot(ma_100_days, 'b')
mlt.plot(data.Close, 'g')

st.pyplot(fig2)

# Moving Average Plot
st.subheader('MOVING-AROUND-200')
ma_200_days = data.Close.rolling(200).mean()
fig3 = mlt.figure(figsize=(10, 8))
mlt.plot(ma_100_days, 'r')
mlt.plot(ma_200_days, 'b')
mlt.plot(data.Close, 'g')

st.pyplot(fig3)

# Prepare test sequences
x = []
y = []
for i in range(100, data_test_scaled.shape[0]):
    x.append(data_test_scaled[i - 100:i])
    y.append(data_test_scaled[i, 0])

x, y = np.array(x), np.array(y)

# Predictions
predict = model.predict(x)

# Inverse scaling
scale = 1/scaler.scale_
predict = predict * scale
y = y * scale


# Moving Final Plot
st.subheader('ORIGINAL-PRICE-PREDICT')
fig4 = mlt.figure(figsize=(10, 8))
mlt.plot(predict, 'r', label="original price")
mlt.plot(y, 'g',label='predicted price')
mlt.xlabel('Time')
mlt.ylabel('Price')

st.pyplot(fig4)
