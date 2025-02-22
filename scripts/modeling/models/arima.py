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
