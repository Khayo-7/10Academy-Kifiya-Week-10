import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scripts.utils.logger import setup_logger

warnings.filterwarnings("ignore")
logger = setup_logger("GARCH_Model")

class GARCHModel:
    """GARCH Model for volatility forecasting with tuning, evaluation, and backtesting."""

    def __init__(self, data: pd.DataFrame, target_col="Price_diff_1", p=1, q=1):
        """
        Initialize the GARCHModel.

        :param data: DataFrame with time series data.
        :param target_col: Column to model (use differenced data for stationarity).
        :param p: Initial order of GARCH terms.
        :param q: Initial order of ARCH terms.
        """
        self.data = data.copy()
        self.target_col = target_col
        self.p = p
        self.q = q
        self.model = None
        self.results = None
        self.best_order = None  # Will be determined by tuning

    def tune_hyperparameters(self, max_p: int = 3, max_q: int = 3):
        """Tunes GARCH hyperparameters using a grid search based on AIC."""
        logger.info("Tuning GARCH hyperparameters...")
        best_aic = float("inf")
        best_order = None
        ts_data = self.data[self.target_col].dropna()

        for p in range(1, max_p + 1):
            for q in range(1, max_q + 1):
                try:
                    model = arch_model(ts_data, vol="Garch", p=p, q=q)
                    fitted_model = model.fit(disp="off")
                    aic = fitted_model.aic

                    if aic < best_aic:
                        best_aic = aic
                        best_order = (p, q)
                except Exception as e:
                    logger.warning(f"Skipping (p={p}, q={q}) due to error: {e}")
                    continue

        self.best_order = best_order if best_order else (1, 1)
        logger.info(f"Selected best GARCH order: {self.best_order}")

    def fit(self):
        """Fits the GARCH model using the best (p, q) order."""
        if not self.best_order:
            raise ValueError("Hyperparameters not tuned. Run tune_hyperparameters() first.")

        ts_data = self.data[self.target_col].dropna()
        logger.info(f"Fitting GARCH{self.best_order} model...")

        self.model = arch_model(ts_data, vol="Garch", p=self.best_order[0], q=self.best_order[1])
        self.results = self.model.fit(disp="off")

        logger.info("GARCH model fitted successfully.")

    def forecast(self, steps: int = 30):
        """Generates volatility forecasts for future periods."""
        if self.results is None:
            raise ValueError("Model is not fitted. Run fit() first.")

        forecast_result = self.results.forecast(horizon=steps)
        return forecast_result.variance.iloc[-1].values  # Extract forecasted variance

    def evaluate(self, test_data: pd.Series):
        """Evaluates the model's performance on test data."""
        if self.results is None:
            raise ValueError("Model not fitted. Run fit() first.")

        # Ensure test_data is numeric
        if not np.issubdtype(test_data.dtype, np.number):
            raise ValueError("test_data must be numeric. Ensure it contains the target variable.")

        # Generate predictions and calculate evaluation metrics
        test_pred = self.results.forecast(start=len(self.data) - len(test_data), horizon=len(test_data)).variance.iloc[:, -1]
        mae = mean_absolute_error(test_data, test_pred)
        rmse = np.sqrt(mean_squared_error(test_data, test_pred))

        return {"MAE": mae, "RMSE": rmse}

    def backtest(self, train_size: float = 0.8) -> dict:
        """Performs backtesting by training on a portion of data and evaluating on the rest."""

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
        # return self.evaluate(test)
        return self.evaluate(test[self.target_col])

    def plot_forecast(self, steps=30):
        """Plots the forecasted volatility alongside historical volatility."""
        forecast = self.forecast(steps)

        # Ensure the index is a datetime index
        if not isinstance(self.data.index, pd.DatetimeIndex):
            self.data.index = pd.to_datetime(self.data.index)

        plt.figure(figsize=(12, 5))
        plt.plot(self.data.index[-100:], self.results.conditional_volatility[-100:], label="Historical Volatility")
        plt.plot(pd.date_range(self.data.index[-1], periods=steps + 1, freq="D")[1:],
                np.sqrt(forecast), label="Forecast Volatility", linestyle="dashed", color="red")
        plt.title(f"GARCH{self.best_order} Volatility Forecast")
        plt.xlabel("Date")
        plt.ylabel("Volatility")
        plt.legend()
        plt.show()
