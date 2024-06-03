import os, shutil
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import streamlit as st

st.set_page_config(page_title='Stock Prediction | App', page_icon=None, layout="centered", initial_sidebar_state="auto", menu_items=None)

st.title("Stock Trend Prediction using LSTM")

st.sidebar.title("")

stock_ticker_input = st.text_input("Enter Stock Ticker", placeholder="e.g. AAPL, GOOG, ^NSEI", value='^NSEI')

st.text("Model trained from 2010-01-01 to 2019-12-31")
start_date = st.text_input("Start Date (YYYY-MM-DD)", placeholder="YYYY-MM-DD",key='start_date', value='2010-01-01')
end_date = st.text_input("End Date (YYYY-MM-DD)", placeholder="YYYY-MM-DD",key='end_date', value='2019-12-31')

if stock_ticker_input and start_date and end_date:
    df = yf.download(stock_ticker_input, start=start_date, end=end_date)
    
    df.reset_index(inplace=True)
    
    st.subheader(f"Data from {start_date} to {end_date}")
    st.dataframe(df.describe(), use_container_width=True)
    
    st.subheader('Closing Price vs Time Chart')
    fig = plt.figure(figsize=(12,6))
    plt.title(f'{stock_ticker_input} Stocks Closing Price ({start_date}-{end_date})')
    plt.plot(df['Date'], df['Close'], 'b', label='Closing price')
    plt.xlabel('Date')
    plt.ylabel('Closing Price in $')
    plt.legend()
    st.pyplot(fig)
    
    ma100 = df.Close.rolling(100).mean()
    
    st.subheader('Closing Price vs Time Chart with 100MA')
    fig = plt.figure(figsize=(12,6))
    plt.title(f'{stock_ticker_input} Stocks Closing Price with 100MA ({start_date}-{end_date})')
    plt.plot(df['Date'], df['Close'], 'b', label='Closing price')
    plt.plot(df['Date'], ma100, 'r', label='100-Day MA')
    plt.xlabel('Date')
    plt.ylabel('Closing Price in $')
    plt.legend()
    st.pyplot(fig)
    
    ma200 = df.Close.rolling(200).mean()
    
    st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
    fig = plt.figure(figsize=(12,6))
    plt.title(f'{stock_ticker_input} Stocks Closing Price with 100MA & 200MA ({start_date}-{end_date})')
    plt.plot(df['Date'], df['Close'], 'b', label='Closing price')
    plt.plot(df['Date'], ma100, 'r', label='100-Day MA')
    plt.plot(df['Date'], ma200, 'g', label='200-Day MA')
    plt.xlabel('Date')
    plt.ylabel('Closing Price in $')
    plt.legend()
    st.pyplot(fig)
    
    data_train = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
    data_test = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])
    
    scaler = MinMaxScaler(feature_range=(0,1))

    past_100_days = data_train.tail(100)
    final_df = pd.concat([past_100_days, data_test], ignore_index=True)
    input_data = scaler.fit_transform(final_df)
    x_test = []
    y_test = []
    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i-100:i])
        y_test.append(input_data[i, 0])
    x_test, y_test = np.array(x_test), np.array(y_test)
    
    st.subheader('Original vs Predictions')
    @st.cache_resource
    def load_stock_model():
        model = load_model(os.path.join(os.getcwd(), 'Stock_LSTM.keras'))
        return model 

    loaded_model = load_stock_model()
    
    y_predicted = loaded_model.predict(x_test)
    scale_factor = 1 / scaler.scale_[0]
    
    y_test = y_test * scale_factor
    y_predicted = y_predicted * scale_factor
    
    fig = plt.figure(figsize=(12,6))
    plt.title(f'{stock_ticker_input} Stock Predictions ({start_date}-{end_date})')
    plt.plot(y_test, 'b', label='Original Price')
    plt.plot(y_predicted, 'r', label='Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Closing Price in $')
    plt.legend()
    st.pyplot(fig)
    
st.subheader('Predict')
start_date_f = st.text_input("Start Date (YYYY-MM-DD)", placeholder="YYYY-MM-DD", key='start_date_f', value='2020-01-01')
end_date_f = st.text_input("End Date (YYYY-MM-DD)", placeholder="YYYY-MM-DD", key='end_date_f', value='2024-01-01')
predict_button = st.button("Predict", use_container_width=True)
    
if predict_button:
    if start_date_f != '' and end_date_f != '':
        new_df = yf.download(stock_ticker_input, start=start_date_f, end=end_date_f)
        input_data = scaler.fit_transform(new_df['Close'].values.reshape(-1, 1))
        
        x_new = []
        for i in range(100, input_data.shape[0]):
            x_new.append(input_data[i-100:i, 0]) 

        x_new = np.array(x_new)
        x_new = np.reshape(x_new, (x_new.shape[0], x_new.shape[1], 1)) 

        y_predicted_f = loaded_model.predict(x_new)
        y_predicted_f = y_predicted_f * scale_factor

        st.subheader('Original vs Predictions')
        fig = plt.figure(figsize=(12,6))
        plt.title(f'{stock_ticker_input} Stock Predictions ({start_date_f}-{end_date_f})')
        plt.plot(new_df.index, new_df['Close'], label='Actual Price')
        plt.plot(new_df.index[100:], y_predicted_f, label='Predicted Price')
        plt.xlabel('Date')
        plt.ylabel('Closing Price in $')
        plt.legend()
        plt.show()
        st.pyplot(fig)
