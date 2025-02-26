import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import pmdarima as pm
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scripts.utils.logger import setup_logger

warnings.filterwarnings("ignore")
logger = setup_logger("ARIMA_Model")

class ARIMAModel:
    def __init__(self, data: pd.DataFrame, target_col: str = "Price", order: tuple = None):
        """ARIMA Model for Univariate Time Series Forecasting with tuning, evaluation, and backtesting."""
        self.data = data.copy()
        self.target_col = target_col
        self.order = order
        self.model = None
        self.results = None
        self.best_order = None

    def tune_hyperparameters(self, seasonal: bool = False, max_p: int = 5, max_d: int = 2, max_q: int = 5):
        """Tunes ARIMA hyperparameters using AutoARIMA."""
        logger.info("Tuning ARIMA hyperparameters using AutoARIMA...")
        auto_arima_model = pm.auto_arima(
            self.data[self.target_col],
            seasonal=seasonal,
            max_p=max_p,
            max_d=max_d,
            max_q=max_q,
            stepwise=True,
            suppress_warnings=True
        )
        self.best_order = auto_arima_model.order
        logger.info(f"Best ARIMA order found: {self.best_order}")

    def fit(self):
        """Fits the ARIMA model using the best order found during tuning."""
        ts_data = self.data[self.target_col].astype(float)

        if self.order is None:
            if not self.best_order:
                raise ValueError("No order provided and hyperparameters not tuned. Run tune_hyperparameters() first.")
            self.order = self.best_order

        logger.info(f"Fitting ARIMA{self.order} model...")
        self.model = ARIMA(ts_data, order=self.order)
        self.results = self.model.fit()

        logger.info("ARIMA model fitted successfully.")

    def forecast(self, steps: int = 30) -> pd.Series:
        """Generates future forecasts."""
        if self.results is None:
            raise ValueError("Model not fitted. Run fit() first.")

        logger.info(f"Generating forecast for {steps} steps ahead...")
        forecast_result = self.results.get_forecast(steps=steps)
        return forecast_result.predicted_mean

    def evaluate(self, test_data: pd.Series) -> dict:
        """Evaluates the model on test data."""
        if self.results is None:
            raise ValueError("Model not fitted. Run fit() first.")

        logger.info("Evaluating ARIMA model performance...")
        test_pred = self.results.get_forecast(steps=len(test_data)).predicted_mean
        mae = mean_absolute_error(test_data, test_pred)
        rmse = np.sqrt(mean_squared_error(test_data, test_pred))

        logger.info(f"Evaluation Results - MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        return {"MAE": mae, "RMSE": rmse}

    def backtest(self, train_size: float = 0.8) -> dict:
        """Performs backtesting on the time series data."""

        if not isinstance(self.data, pd.DataFrame):
            raise ValueError("self.data must be a DataFrame with a numeric target column.")
        
        split_idx = int(len(self.data) * train_size)
        # train, test = self.data[:split_idx], self.data[split_idx:]
        train, test = self.data.iloc[:split_idx], self.data.iloc[split_idx:]

        # Tune hyperparameters and fit the model
        logger.info(f"Starting backtest: Train Size = {train_size * 100:.1f}%, Test Size = {(1 - train_size) * 100:.1f}%")
        self.data = train
        self.tune_hyperparameters()
        self.fit()

        # Evaluate the model on the test data
        return self.evaluate(test[self.target_col])

    def plot_forecast(self, steps: int = 30):
        """Plot forecast against historical data."""
        forecast = self.forecast(steps)

        # Ensure the index is a datetime index
        if not isinstance(self.data.index, pd.DatetimeIndex):
            self.data.index = pd.to_datetime(self.data.index)
            
        plt.figure(figsize=(12, 5))
        plt.plot(self.data.index, self.data[self.target_col], label="Actual")
        plt.plot(pd.date_range(self.data.index[-1], periods=steps + 1, freq="D")[1:],
                forecast, label="Forecast", linestyle="dashed", color="red")
        plt.title(f"ARIMA{self.order} Forecast")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.show()

# import numpy as np
# import pandas as pd
# import statsmodels.api as sm
# import matplotlib.pyplot as plt
# import seaborn as sns
# from statsmodels.tsa.stattools import adfuller
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# from pmdarima import auto_arima
# from sklearn.metrics import mean_absolute_error, mean_squared_error

# class ARIMAModel:
#     def __init__(self, data_path: str, date_col: str = "Date", price_col: str = "Price"):
#         self.data_path = data_path
#         self.date_col = date_col
#         self.price_col = price_col
#         self.data = None
#         self.model = None

#     def load_data(self):
#         """Loads and preprocesses data."""
#         df = pd.read_csv(self.data_path, parse_dates=[self.date_col])
#         df.set_index(self.date_col, inplace=True)
#         df = df[[self.price_col]].dropna()
#         self.data = df
#         return df

#     def check_stationarity(self):
#         """Performs ADF test and returns p-value."""
#         result = adfuller(self.data[self.price_col])
#         print(f"ADF Statistic: {result[0]}")
#         print(f"p-value: {result[1]}")
#         return result[1] < 0.05  # If p < 0.05, data is stationary

#     def difference_data(self):
#         """Applies differencing if data is non-stationary."""
#         self.data["diff_price"] = self.data[self.price_col].diff().dropna()
#         return self.data.dropna()

#     def plot_acf_pacf(self):
#         """Plots ACF and PACF to determine ARIMA parameters."""
#         fig, axes = plt.subplots(1, 2, figsize=(12, 5))
#         plot_acf(self.data[self.price_col].dropna(), ax=axes[0])
#         plot_pacf(self.data[self.price_col].dropna(), ax=axes[1])
#         plt.show()

#     def optimize_arima(self):
#         """Uses Auto-ARIMA to find optimal parameters."""
#         model = auto_arima(self.data[self.price_col], seasonal=False, trace=True)
#         print(f"Optimal ARIMA order: {model.order}")
#         return model.order

#     def train_model(self, order):
#         """Trains the ARIMA model."""
#         self.model = sm.tsa.ARIMA(self.data[self.price_col], order=order)
#         self.model = self.model.fit()
#         print(self.model.summary())

#     def evaluate_model(self):
#         """Evaluates ARIMA model on test data."""
#         train_size = int(len(self.data) * 0.8)
#         train, test = self.data.iloc[:train_size], self.data.iloc[train_size:]

#         model = sm.tsa.ARIMA(train[self.price_col], order=self.model.model.order)
#         model_fit = model.fit()

#         predictions = model_fit.forecast(steps=len(test))
#         test["Predictions"] = predictions

#         mae = mean_absolute_error(test[self.price_col], test["Predictions"])
#         rmse = np.sqrt(mean_squared_error(test[self.price_col], test["Predictions"]))

#         print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")

#         plt.figure(figsize=(12, 5))
#         plt.plot(train[self.price_col], label="Train")
#         plt.plot(test[self.price_col], label="Test", color="blue")
#         plt.plot(test["Predictions"], label="Predictions", color="red")
#         plt.legend()
#         plt.show()

#     def forecast(self, steps=30):
#         """Generates future price predictions."""
#         forecast = self.model.forecast(steps=steps)
#         plt.figure(figsize=(10, 5))
#         plt.plot(self.data[self.price_col], label="Historical Data")
#         plt.plot(pd.date_range(self.data.index[-1], periods=steps, freq="D"), forecast, label="Forecast", color="red")
#         plt.legend()
#         plt.show()
#         return forecast
