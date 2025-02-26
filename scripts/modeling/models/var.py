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

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from statsmodels.tsa.api import VAR
# from statsmodels.tsa.stattools import adfuller
# from statsmodels.stats.diagnostic import acorr_ljungbox
# from scipy.stats import shapiro

# class VARModel:
#     def __init__(self, data_path, date_col="Date"):
#         self.data_path = data_path
#         self.date_col = date_col
#         self.data = None
#         self.model = None

#     def load_data(self):
#         """Loads and preprocesses data."""
#         df = pd.read_csv(self.data_path, parse_dates=[self.date_col])
#         df.set_index(self.date_col, inplace=True)
#         df.dropna(inplace=True)
        
#         # Convert to percentage change for stationarity
#         self.data = df.pct_change().dropna()
#         return self.data

#     def check_stationarity(self):
#         """Performs ADF test for each variable and returns results."""
#         results = {}
#         for col in self.data.columns:
#             result = adfuller(self.data[col])
#             results[col] = {
#                 "ADF Statistic": result[0],
#                 "p-value": result[1],
#                 "Stationary": result[1] < 0.05
#             }
#         return results

#     def select_optimal_lags(self, max_lags=15):
#         """Determines optimal lag order using AIC and BIC criteria."""
#         model = VAR(self.data)
#         lag_selection = model.select_order(maxlags=max_lags)
#         return lag_selection.aic, lag_selection.bic, lag_selection.fpe

#     def fit_var(self, lags):
#         """Fits VAR model with selected lag order."""
#         model = VAR(self.data)
#         self.model = model.fit(lags)
#         print(self.model.summary())

#     def granger_causality_test(self, target_var):
#         """Performs Granger causality test on all variables for a given target variable."""
#         from statsmodels.tsa.stattools import grangercausalitytests
#         results = {}
#         for col in self.data.columns:
#             if col != target_var:
#                 test_result = grangercausalitytests(self.data[[col, target_var]], maxlag=3, verbose=False)
#                 p_values = [test_result[i+1][0]['ssr_chi2test'][1] for i in range(3)]
#                 results[col] = min(p_values)  # Choose lowest p-value
#         return results

#     def plot_impulse_response(self, steps=20):
#         """Plots impulse response function for all variables."""
#         irf = self.model.irf(steps)
#         irf.plot(figsize=(12, 6))
#         plt.show()

#     def forecast(self, steps=10):
#         """Forecasts future values using VAR model."""
#         forecast_input = self.data.values[-self.model.k_ar:]
#         forecast = self.model.forecast(forecast_input, steps=steps)
        
#         # Convert to DataFrame
#         forecast_df = pd.DataFrame(forecast, index=pd.date_range(start=self.data.index[-1], periods=steps+1, freq='D')[1:], columns=self.data.columns)
        
#         forecast_df.plot(figsize=(12, 6))
#         plt.title("VAR Forecast")
#         plt.show()
        
#         return forecast_df
