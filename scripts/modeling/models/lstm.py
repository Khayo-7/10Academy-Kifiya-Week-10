import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scripts.utils.logger import setup_logger

warnings.filterwarnings("ignore")
logger = setup_logger("LSTM_Model")

class LSTMModel:
    """LSTM Model for time series forecasting with training, evaluation, and backtesting."""

    def __init__(self, data: pd.DataFrame, target_col, lookback=30, neurons=50, dropout=0.2, epochs=50, batch_size=32):
        """
        Initializes the LSTM model.

        :param data: DataFrame containing time series data.
        :param target_col: Column name of target variable.
        :param lookback: Number of past time steps to use for predictions.
        :param neurons: Number of neurons in LSTM layer.
        :param dropout: Dropout rate for regularization.
        :param epochs: Number of training epochs.
        :param batch_size: Training batch size.
        """
        self.data = data.copy()
        self.target_col = target_col
        self.lookback = lookback
        self.neurons = neurons
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def preprocess_data(self):
        """Scales and prepares sequences for LSTM training."""
        logger.info("Preprocessing data...")
        scaled_data = self.scaler.fit_transform(self.data[[self.target_col]])

        X, y = [], []
        for i in range(self.lookback, len(scaled_data)):
            X.append(scaled_data[i - self.lookback:i])
            y.append(scaled_data[i])

        self.X_train, self.y_train = np.array(X), np.array(y)
        logger.info(f"Training data shape: X={self.X_train.shape}, y={self.y_train.shape}")

    def build_model(self):
        """Defines the LSTM model architecture."""
        logger.info("Building LSTM model...")
        model = Sequential([
            LSTM(self.neurons, return_sequences=True, input_shape=(self.lookback, 1)),
            Dropout(self.dropout),
            LSTM(self.neurons, return_sequences=False),
            Dropout(self.dropout),
            Dense(1)
        ])
        model.compile(optimizer=Adam(), loss="mse")
        self.model = model
        logger.info("LSTM model built successfully.")

    def train(self):
        """Trains the LSTM model."""
        if self.model is None:
            logger.error("Model not built. Run `build_model()` first.")
            raise ValueError("Run `build_model()` before training.")

        logger.info(f"Training LSTM for {self.epochs} epochs...")
        self.model.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=1)

    def forecast(self, steps=30):
        """Generates forecasts for future periods."""
        if self.model is None:
            logger.error("Model not trained. Run `train()` first.")
            raise ValueError("Run `train()` before forecasting.")

        logger.info(f"Forecasting {steps} steps ahead...")
        last_sequence = self.X_train[-1]  # Last known sequence
        predictions = []

        for _ in range(steps):
            next_pred = self.model.predict(last_sequence.reshape(1, self.lookback, 1))[0]
            predictions.append(next_pred)
            last_sequence = np.vstack([last_sequence[1:], next_pred])

        forecast_values = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        forecast_df = pd.DataFrame(forecast_values, columns=[self.target_col])

        return forecast_df

    def evaluate(self, actual: pd.DataFrame, predicted: pd.DataFrame):
        """
        Evaluates the model using MAE, RMSE, and MAPE.

        :param actual: DataFrame of actual values.
        :param predicted: DataFrame of predicted values.
        :return: Dictionary of evaluation metrics.
        """
        if actual.shape != predicted.shape:
            logger.error("Actual and predicted dataframes must have the same shape.")
            raise ValueError("Mismatched shapes between actual and predicted.")

        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100

        metrics = {"MAE": mae, "RMSE": rmse, "MAPE": mape}
        logger.info(f"Evaluation Metrics: {metrics}")

        return metrics

    def backtest(self, train_size=0.8):
        """
        Performs backtesting using a rolling window approach.

        :param train_size: Proportion of data to use for training.
        :return: Backtest results including evaluation metrics.
        """
        logger.info("Starting backtest...")
        split_idx = int(len(self.data) * train_size)
        train, test = self.data.iloc[:split_idx], self.data.iloc[split_idx:]

        self.scaler.fit(train[[self.target_col]])  # Refit scaler on training data
        scaled_train = self.scaler.transform(train[[self.target_col]])
        scaled_test = self.scaler.transform(test[[self.target_col]])

        history = list(scaled_train.flatten())
        actuals, predictions = [], []

        for t in range(len(scaled_test)):
            X_input = np.array(history[-self.lookback:]).reshape(1, self.lookback, 1)
            pred = self.model.predict(X_input)[0]
            actuals.append(scaled_test[t])
            predictions.append(pred)
            history.append(scaled_test[t])  # Append actual value

        actual_df = pd.DataFrame(self.scaler.inverse_transform(np.array(actuals).reshape(-1, 1)), index=test.index, columns=[self.target_col])
        predicted_df = pd.DataFrame(self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1)), index=test.index, columns=[self.target_col])

        return self.evaluate(actual_df, predicted_df)

    def plot_forecast(self, steps=30):
        """Plots actual vs. forecasted values."""
        forecast = self.forecast(steps)

        # Ensure the index is a datetime index
        if not isinstance(self.data.index, pd.DatetimeIndex):
            self.data.index = pd.to_datetime(self.data.index)

        plt.figure(figsize=(12, 6))
        plt.plot(self.data.index, self.data[self.target_col], label="Actual", linewidth=2)
        plt.plot(pd.date_range(self.data.index[-1], periods=steps, freq='D'),
                forecast[self.target_col], label="Forecast", linestyle="dashed")

        plt.legend()
        plt.title("LSTM Forecast")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.show()

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout

# class LSTMModel:
#     def __init__(self, data_path, date_col="Date", price_col="Price", seq_length=60):
#         self.data_path = data_path
#         self.date_col = date_col
#         self.price_col = price_col
#         self.seq_length = seq_length
#         self.scaler = MinMaxScaler(feature_range=(0, 1))
#         self.model = None

#     def load_data(self):
#         """Loads and normalizes Brent oil price data."""
#         df = pd.read_csv(self.data_path, parse_dates=[self.date_col])
#         df.set_index(self.date_col, inplace=True)
#         df.dropna(inplace=True)

#         # Normalize price
#         df[self.price_col] = self.scaler.fit_transform(df[[self.price_col]])
#         self.data = df
#         return df

#     def create_sequences(self):
#         """Generates input sequences and corresponding labels for training."""
#         data = self.data[self.price_col].values
#         X, y = [], []
#         for i in range(len(data) - self.seq_length):
#             X.append(data[i:i + self.seq_length])
#             y.append(data[i + self.seq_length])
#         return np.array(X), np.array(y)

#     def build_model(self):
#         """Defines and compiles the LSTM model."""
#         self.model = Sequential([
#             LSTM(50, return_sequences=True, input_shape=(self.seq_length, 1)),
#             Dropout(0.2),
#             LSTM(50, return_sequences=False),
#             Dropout(0.2),
#             Dense(25),
#             Dense(1)
#         ])
#         self.model.compile(optimizer="adam", loss="mean_squared_error")

#     def train_model(self, X_train, y_train, epochs=50, batch_size=32):
#         """Trains the LSTM model."""
#         self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

#     def forecast(self, last_sequence, steps=30):
#         """Generates forecasts based on the trained model."""
#         predictions = []
#         current_seq = last_sequence

#         for _ in range(steps):
#             pred = self.model.predict(current_seq.reshape(1, self.seq_length, 1))[0, 0]
#             predictions.append(pred)
#             current_seq = np.append(current_seq[1:], pred).reshape(self.seq_length, 1)

#         return self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    
#     def plot_forecast(self, actual_prices, predictions):
#         """Plots actual vs. predicted prices."""
#         plt.figure(figsize=(12, 6))
#         plt.plot(actual_prices, label="Actual Prices")
#         plt.plot(predictions, label="LSTM Forecast", linestyle="dashed")
#         plt.legend()
#         plt.title("Brent Oil Price Prediction with LSTM")
#         plt.show()
