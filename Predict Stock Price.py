# AI-Driven Stock Price Prediction using Time Series Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings
warnings.filterwarnings("ignore")

# 1.Load and preprocess data
df = pd.read_csv("AI.csv")
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values('Date', inplace=True)
df.reset_index(drop=True, inplace=True)

# 2.Plot closing price
plt.figure(figsize=(10, 5))
plt.plot(df['Date'], df['Close'], label='Close Price')
plt.title("Stock Closing Price")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()

# 3.ARIMA Model
arima_data = df[['Date', 'Close']].copy().set_index('Date')
train, test = arima_data[:-20], arima_data[-20:]
arima = ARIMA(train, order=(5,1,0)).fit()
arima_pred = arima.forecast(steps=20)
print("\nARIMA RMSE:", np.sqrt(mean_squared_error(test, arima_pred)))
print("ARIMA MAE:", mean_absolute_error(test, arima_pred))

# 4.Prophet Model
prophet_df = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
prophet = Prophet()
prophet.fit(prophet_df)
future = prophet.make_future_dataframe(periods=20)
forecast = prophet.predict(future)
prophet.plot(forecast)
plt.title("Prophet Forecast")
plt.show()

# 5.LSTM Model
scaled = MinMaxScaler().fit_transform(df[['Close']])
X, y = [], []
for i in range(60, len(scaled)):
    X.append(scaled[i-60:i, 0])
    y.append(scaled[i, 0])
X, y = np.array(X), np.array(y)
X = X.reshape((X.shape[0], X.shape[1], 1))

lstm = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])
lstm.compile(optimizer='adam', loss='mean_squared_error')
lstm.fit(X, y, epochs=20, batch_size=32, verbose=0)

# 6.LSTM Prediction
X_test = [scaled[-80+i-60: -80+i] for i in range(60, 80)]
X_test = np.reshape(X_test, (20, 60, 1))
lstm_pred = lstm.predict(X_test)
lstm_pred = MinMaxScaler().fit(df[['Close']]).inverse_transform(lstm_pred)
actual = df['Close'].values[-20:]
print("\nLSTM RMSE:", np.sqrt(mean_squared_error(actual, lstm_pred)))
print("LSTM MAE:", mean_absolute_error(actual, lstm_pred))

# 7.Comparison Plot
plt.figure(figsize=(10, 5))
plt.plot(actual, label="Actual")
plt.plot(lstm_pred, label="LSTM Predicted")
plt.title("LSTM Forecast vs Actual")
plt.legend()
plt.grid(True)
plt.show()
