import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pmdarima import auto_arima
from scripts.utils.logger import setup_logger

warnings.filterwarnings("ignore")
logger = setup_logger("MS_ARIMA_Model")


class MSARIMAModel:
    """Markov-Switching ARIMA model for regime-based time series forecasting."""

    def __init__(self, data: pd.DataFrame, target_col="Price", num_regimes=2, p=1, d=1, q=1):
        """
        Initialize the MS-ARIMA Model.

        :param data: DataFrame with time series data.
        :param target_col: Column to model.
        :param num_regimes: Number of regimes in the Markov model.
        :param p: AR order.
        :param d: Differencing order.
        :param q: MA order.
        """
        self.data = data.copy()
        self.target_col = target_col
        self.num_regimes = num_regimes
        self.p = p
        self.d = d
        self.q = q
        self.best_order = None
        self.best_regimes = None
        self.model = None
        self.results = None

    def tune_hyperparameters(self, max_p=3, max_d=2, max_q=3, max_regimes=3):
        """Tunes ARIMA orders (p, d, q) and the number of regimes (k) using AIC."""
        logger.info("Tuning MS-ARIMA hyperparameters...")
        ts_data = self.data[self.target_col].dropna()

        # Auto-tune ARIMA order using pmdarima
        try:
            arima_model = auto_arima(ts_data, max_p=max_p, max_d=max_d, max_q=max_q,
                                    seasonal=False, stepwise=True, suppress_warnings=True, trace=False)
            best_arima_order = arima_model.order
            logger.info(f"Optimal ARIMA order: {best_arima_order}")
        except Exception as e:
            logger.error(f"Auto-ARIMA tuning failed: {e}")
            best_arima_order = (1, 1, 1)  # Fallback order

        # Tune the number of regimes
        best_aic = float("inf")
        best_regimes = None

        for k in range(2, max_regimes + 1):
            try:
                model = MarkovRegression(ts_data, k_regimes=k, trend="c", switching_variance=True)
                fitted_model = model.fit()
                aic = fitted_model.aic

                if aic < best_aic:
                    best_aic = aic
                    best_regimes = k
            except Exception as e:
                logger.warning(f"Skipping k={k} due to error: {e}")
                continue

        self.best_order = best_arima_order
        self.best_regimes = best_regimes if best_regimes else 2  # Default to 2 regimes if tuning fails
        logger.info(f"Selected best number of regimes: {self.best_regimes}")

    def fit(self):
        """Fits the MS-ARIMA model using the best hyperparameters found."""
        if not self.best_order or not self.best_regimes:
            raise ValueError("Hyperparameters not tuned. Run tune_hyperparameters() first.")

        ts_data = self.data[self.target_col].dropna()
        logger.info(f"Fitting MS-ARIMA({self.best_order}, {self.best_regimes} regimes) model...")

        self.model = MarkovRegression(ts_data, k_regimes=self.best_regimes, trend="c", switching_variance=True)
        self.results = self.model.fit()

        logger.info("MS-ARIMA model fitted successfully.")

    def forecast(self, steps=30):
        """Generates future predictions using the fitted model."""
        if self.results is None:
            raise ValueError("Model is not fitted. Run fit() first.")

        forecast = self.results.predict(start=len(self.data), end=len(self.data) + steps - 1)
        return forecast.values

    def evaluate(self, test_data: pd.Series):
        """Evaluates the model on test data using MAE and RMSE."""
        if self.results is None:
            raise ValueError("Model not fitted. Run fit() first.")

        test_pred = self.results.predict(start=len(self.data) - len(test_data), end=len(self.data) - 1)
        mae = mean_absolute_error(test_data, test_pred)
        rmse = np.sqrt(mean_squared_error(test_data, test_pred))

        return {"MAE": mae, "RMSE": rmse}

    def backtest(self, train_size: float = 0.8) -> dict:
        """Performs backtesting by splitting the dataset into train/test and evaluating the model."""

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
        """Plots the forecasted values alongside historical data."""
        forecast = self.forecast(steps)

        # Ensure the index is a datetime index
        if not isinstance(self.data.index, pd.DatetimeIndex):
            self.data.index = pd.to_datetime(self.data.index)

        plt.figure(figsize=(12, 5))
        plt.plot(self.data.index[-100:], self.data[self.target_col].iloc[-100:], label="Historical Data")
        plt.plot(pd.date_range(self.data.index[-1], periods=steps + 1, freq="D")[1:],
                forecast, label="Forecast", linestyle="dashed", color="red")
        plt.title(f"MS-ARIMA({self.best_order}) Forecast ({self.best_regimes} Regimes)")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend()
        plt.show()
