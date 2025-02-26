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

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from arch import arch_model
# from statsmodels.tsa.stattools import adfuller

# class GARCHModel:
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
        
#         # Convert to returns (log differences)
#         df["returns"] = np.log(df[self.price_col] / df[self.price_col].shift(1))
#         df.dropna(inplace=True)
        
#         self.data = df
#         return df

#     def check_stationarity(self):
#         """Performs ADF test and returns p-value."""
#         result = adfuller(self.data["returns"])
#         print(f"ADF Statistic: {result[0]}")
#         print(f"p-value: {result[1]}")
#         return result[1] < 0.05  # If p < 0.05, data is stationary

#     def plot_volatility(self):
#         """Plots returns to check for volatility clustering."""
#         plt.figure(figsize=(12, 5))
#         plt.plot(self.data["returns"], label="Returns", color="blue")
#         plt.title("Brent Oil Returns with Volatility Clustering")
#         plt.legend()
#         plt.show()

#     def fit_garch(self, p=1, q=1):
#         """Fits a GARCH model with given parameters."""
#         model = arch_model(self.data["returns"], vol="Garch", p=p, q=q)
#         self.model = model.fit(disp="off")
#         print(self.model.summary())

#     def plot_conditional_volatility(self):
#         """Plots the estimated volatility."""
#         self.data["Volatility"] = self.model.conditional_volatility
#         plt.figure(figsize=(12, 5))
#         plt.plot(self.data["Volatility"], label="Estimated Volatility", color="red")
#         plt.title("GARCH Estimated Volatility")
#         plt.legend()
#         plt.show()

#     def forecast_volatility(self, steps=30):
#         """Forecasts future volatility."""
#         forecast = self.model.forecast(horizon=steps)
#         forecast_vol = forecast.variance[-1, :]
        
#         plt.figure(figsize=(10, 5))
#         plt.plot(forecast_vol, label="Forecasted Volatility", color="green")
#         plt.title("GARCH Forecasted Volatility")
#         plt.legend()
#         plt.show()
        
#         return forecast_vol

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from arch import arch_model

# class GARCHModel:
#     def __init__(self, data_path, date_col="Date", price_col="Price"):
#         self.data_path = data_path
#         self.date_col = date_col
#         self.price_col = price_col
#         self.model = None
#         self.fitted_model = None

#     def load_data(self):
#         """Loads Brent oil price data and computes log returns."""
#         df = pd.read_csv(self.data_path, parse_dates=[self.date_col])
#         df.set_index(self.date_col, inplace=True)
#         df.dropna(inplace=True)

#         # Compute log returns
#         df["log_return"] = np.log(df[self.price_col] / df[self.price_col].shift(1))
#         df.dropna(inplace=True)
#         self.data = df
#         return df

#     def fit_model(self, p=1, q=1):
#         """Fits a GARCH(p, q) model to log returns."""
#         returns = self.data["log_return"] * 100  # Scale returns for stability
#         self.model = arch_model(returns, vol="Garch", p=p, q=q, mean="Zero", dist="Normal")
#         self.fitted_model = self.model.fit(disp="off")  # Disable verbose output
#         return self.fitted_model.summary()

#     def forecast_volatility(self, days=30):
#         """Generates volatility forecasts for the specified horizon."""
#         if self.fitted_model is None:
#             raise ValueError("Model is not trained. Call fit_model() first.")

#         forecasts = self.fitted_model.forecast(horizon=days)
#         return forecasts.variance[-1:].T  # Return variance predictions

#     def plot_volatility(self):
#         """Plots actual vs. predicted volatility."""
#         plt.figure(figsize=(12, 6))
#         plt.plot(self.data.index[-500:], self.fitted_model.conditional_volatility[-500:], label="GARCH Volatility")
#         plt.title("Brent Oil Price Volatility (GARCH Model)")
#         plt.xlabel("Date")
#         plt.ylabel("Volatility")
#         plt.legend()
#         plt.show()

# import numpy as np
# import pandas as pd
# from arch import arch_model
# from sklearn.metrics import mean_squared_error

# class GARCHBacktester:
#     def __init__(self, data_path, window_size=500, p=1, q=1):
#         self.data_path = data_path
#         self.window_size = window_size
#         self.p = p
#         self.q = q
#         self.data = None
#         self.predictions = []
#         self.actual_volatility = []

#     def load_data(self):
#         """Loads Brent oil price data and computes log returns."""
#         df = pd.read_csv(self.data_path, parse_dates=["Date"])
#         df.set_index("Date", inplace=True)
#         df.dropna(inplace=True)

#         # Compute log returns
#         df["log_return"] = np.log(df["Price"] / df["Price"].shift(1))
#         df.dropna(inplace=True)
#         self.data = df
#         return df

#     def backtest(self):
#         """Performs rolling window backtesting of the GARCH model."""
#         log_returns = self.data["log_return"] * 100  # Scale for stability
#         total_points = len(log_returns)

#         for start in range(total_points - self.window_size):
#             train = log_returns[start : start + self.window_size]

#             # Fit GARCH model
#             model = arch_model(train, vol="Garch", p=self.p, q=self.q, mean="Zero", dist="Normal")
#             fitted_model = model.fit(disp="off")

#             # Predict next-day volatility
#             forecast = fitted_model.forecast(start=len(train), horizon=1)
#             predicted_vol = np.sqrt(forecast.variance.iloc[-1, 0])  # Convert variance to volatility
#             actual_vol = np.abs(log_returns.iloc[start + self.window_size])

#             # Store results
#             self.predictions.append(predicted_vol)
#             self.actual_volatility.append(actual_vol)

#         return self.predictions, self.actual_volatility

#     def evaluate(self):
#         """Calculates RMSE to measure prediction accuracy."""
#         return np.sqrt(mean_squared_error(self.actual_volatility, self.predictions))
