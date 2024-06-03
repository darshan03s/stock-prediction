import os, shutil
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from IPython.display import display, HTML
def disp(obj):
    display(HTML(obj.to_html()))
start = '2010-01-01'
end = '2019-12-31'
ticker_symbol = 'AAPL'
df = yf.download(ticker_symbol, start=start, end=end)
disp(df.head())
display(HTML(df.tail().to_html()))
df = df.reset_index()
display(HTML(df.head().to_html()))
df.columns.tolist()
plt.plot(df['Date'], df['Close'])
plt.title(ticker_symbol+'_Stocks Closing Price')
plt.xlabel('Date')
plt.ylabel('Closing Price in $')
correlation_matrix = df.corr()
display(HTML(correlation_matrix.to_html()))
ma100 = df.Close.rolling(100).mean()
ma100
plt.figure(figsize=(12,6))
plt.plot(df['Date'], df['Close'])
plt.plot(df['Date'], ma100, 'r')
plt.title(ticker_symbol+'_Stocks')
plt.xlabel('Date')
plt.ylabel('Price in $')
ma200 = df.Close.rolling(200).mean()
ma200
plt.figure(figsize=(12,6))
plt.plot(df['Date'], df['Close'])
plt.plot(df['Date'], ma100, 'r')
plt.plot(df['Date'], ma200, 'g')
plt.title(ticker_symbol+'_Stocks')
plt.xlabel('Date')
plt.ylabel('Price in $')
df.shape
data_train = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_test = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])
data_train.shape, data_test.shape
data_train.head()
data_test.head()
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
data_train_arr = scaler.fit_transform(data_train)
data_train_arr
data_train_arr.shape
x_train = []
y_train = []
for i in range(100, data_train_arr.shape[0]):
    x_train.append(data_train_arr[i-100:i])
    y_train.append(data_train_arr[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)
x_train.shape
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
model = Sequential()
model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=60, activation='relu', return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(units=80, activation='relu', return_sequences=True))
model.add(Dropout(0.4))
model.add(LSTM(units=120, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()
model.fit(x_train, y_train, epochs=50)
model.save('Stock_LSTM.keras')
past_100_days = data_train.tail(100)
final_df = pd.concat([past_100_days, data_test], ignore_index=True)
final_df
input_data = scaler.fit_transform(final_df)
input_data
input_data.shape
x_test = []
y_test = []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])
x_test, y_test = np.array(x_test), np.array(y_test)
x_test.shape, y_test.shape
y_predicted = model.predict(x_test)
y_predicted.shape
scaler.scale_
scale_factor = 1 / scaler.scale_[0]
y_test = y_test * scale_factor
y_predicted = y_predicted * scale_factor
plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.title(ticker_symbol+'_Stocks Predictions')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
from tensorflow.keras.models import load_model
loaded_model = load_model('/kaggle/working/Stock_LSTM.keras')
today = datetime.today().strftime('%Y-%m-%d')
ticker_symbol = 'AAPL'
new_df = yf.download(ticker_symbol, start='2020-01-01', end=today)
new_df
new_df.shape
input_data = scaler.fit_transform(new_df['Close'].values.reshape(-1, 1))
x_new = []
for i in range(100, input_data.shape[0]):
    x_new.append(input_data[i-100:i, 0])
x_new = np.array(x_new)
y_predicted = loaded_model.predict(x_new)
y_predicted = y_predicted * scale_factor
plt.figure(figsize=(12, 6))
plt.plot(new_df.index, new_df['Close'], label='Actual Price')
plt.plot(new_df.index[100:], y_predicted, label='Predicted Price')
plt.title(f'{ticker_symbol} Future Stock Predictions')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
