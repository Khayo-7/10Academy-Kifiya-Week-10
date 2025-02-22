import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scripts.utils.logger import setup_logger

warnings.filterwarnings("ignore")
logger = setup_logger("VAR_Model")

class VARModel:
    """VAR Model with tuning, backtesting, evaluation, and visualization."""

    def __init__(self, data: pd.DataFrame, target_cols=None, max_lags=10):
        """
        Initializes the VAR model.

        :param data: DataFrame containing time series data.
        :param target_cols: List of columns to include in VAR model.
        :param max_lags: Maximum lag order to consider.
        """
        self.data = data.copy()
        self.target_cols = target_cols if target_cols else data.columns.tolist()
        self.max_lags = max_lags
        self.lags = None
        self.model = None

    def tune_hyperparameters(self):
        """Selects the optimal lag order based on Akaike Information Criterion (AIC)."""
        logger.info("Selecting optimal lag length...")
        model = VAR(self.data[self.target_cols])
        lag_selection = model.select_order(self.max_lags)
        self.lags = lag_selection.aic
        logger.info(f"Optimal lags selected: {self.lags}")

    def fit(self):
        """Fits the VAR model using the best lag order."""
        if self.lags is None:
            logger.error("Lags not selected. Run `tune_hyperparameters()` first.")
            raise ValueError("Run `tune_hyperparameters()` before fitting the model.")

        logger.info(f"Fitting VAR model with {self.lags} lags...")
        self.model = VAR(self.data[self.target_cols]).fit(self.lags)

    def forecast(self, steps=30):
        """Generates forecasts for future periods."""
        if self.model is None:
            logger.error("Model not trained. Run `fit()` first.")
            raise ValueError("Run `fit()` before forecasting.")

        logger.info(f"Forecasting {steps} steps ahead...")
        forecast_values = self.model.forecast(self.data[self.target_cols].values[-self.lags:], steps)
        forecast_df = pd.DataFrame(forecast_values, columns=self.target_cols)

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

        history = train[self.target_cols].values
        actuals, predictions = [], []

        for t in range(len(test)):
            model = VAR(history).fit(self.lags)
            forecast = model.forecast(history[-self.lags:], steps=1)
            actuals.append(test[self.target_cols].iloc[t].values)
            predictions.append(forecast[0])
            history = np.vstack([history, test[self.target_cols].iloc[t].values])

        actual_df = pd.DataFrame(actuals, columns=self.target_cols, index=test.index)
        predicted_df = pd.DataFrame(predictions, columns=self.target_cols, index=test.index)

        return self.evaluate(actual_df, predicted_df)

    def plot_forecast(self, steps=30):
        """Plots actual vs. forecasted values."""
        forecast = self.forecast(steps)

        # Ensure the index is a datetime index
        if not isinstance(self.data.index, pd.DatetimeIndex):
            self.data.index = pd.to_datetime(self.data.index)

        plt.figure(figsize=(12, 6))

        for col in self.target_cols:
            plt.plot(self.data.index, self.data[col], label=f"Actual {col}", linewidth=2)
            plt.plot(pd.date_range(self.data.index[-1], periods=steps, freq='D'),
                    forecast[col], label=f"Forecast {col}", linestyle="dashed")

        plt.legend()
        plt.title("VAR Forecast")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.show()
