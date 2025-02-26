import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from scripts.utils.logger import setup_logger
import pyflux as pf

warnings.filterwarnings("ignore")
logger = setup_logger("MGARCH_Model")

class MGARCHModel:
    """MGARCH Model for multivariate volatility forecasting with BEKK/DCC variants."""

    def __init__(self, data: pd.DataFrame, target_cols, model_type="DCC", p=1, q=1):
        """
        Initializes the MGARCH model.

        :param data: DataFrame containing target price data.
        :param target_cols: List of target column names.
        :param model_type: Type of MGARCH model ("BEKK" or "DCC").
        :param p: Order of the ARCH term.
        :param q: Order of the GARCH term.
        """
        self.data = data.copy()
        self.target_cols = target_cols
        self.model_type = model_type
        self.p = p
        self.q = q
        self.returns = None
        self.model = None
        self.fitted_model = None

    def preprocess_data(self):
        """Calculates log returns and checks stationarity."""
        logger.info("Preprocessing data...")
        self.returns = np.log(self.data[self.target_cols] / self.data[self.target_cols].shift(1)).dropna()
        
        for target in self.target_cols:
            adf_result = adfuller(self.returns[target])
            logger.info(f"ADF Test for {target}: p-value = {adf_result[1]} (Stationary if < 0.05)")

    def build_model(self):
        """Defines the MGARCH model based on the specified type."""
        if self.model_type == "DCC":
            logger.info("Building DCC-MGARCH model...")
            self.model = pf.DCCGARCH(self.returns, p=self.p, q=self.q)
        elif self.model_type == "BEKK":
            logger.info("Building BEKK-MGARCH model...")
            self.model = pf.BEKKGARCH(self.returns, p=self.p, q=self.q)
        else:
            logger.error("Invalid model type. Choose 'DCC' or 'BEKK'.")
            raise ValueError("Invalid MGARCH model type.")

    def train(self):
        """Trains the MGARCH model."""
        if self.model is None:
            logger.error("Model not built. Run `build_model()` first.")
            raise ValueError("Run `build_model()` before training.")

        logger.info("Training MGARCH model...")
        self.fitted_model = self.model.fit()
        logger.info(f"Model training completed. AIC: {self.fitted_model.aic}, BIC: {self.fitted_model.bic}")

    def forecast(self, steps=30):
        """Generates future volatility forecasts."""
        if self.fitted_model is None:
            logger.error("Model not trained. Run `train()` first.")
            raise ValueError("Run `train()` before forecasting.")

        logger.info(f"Forecasting {steps} steps ahead...")
        forecasts = self.fitted_model.predict(h=steps)
        return forecasts

    def evaluate(self):
        """Evaluates the MGARCH model."""
        if self.fitted_model is None:
            logger.error("Model not trained. Run `train()` first.")
            raise ValueError("Run `train()` before evaluation.")

        log_likelihood = self.fitted_model.log_likelihood
        aic = self.fitted_model.aic
        bic = self.fitted_model.bic

        metrics = {"Log-Likelihood": log_likelihood, "AIC": aic, "BIC": bic}
        logger.info(f"Evaluation Metrics: {metrics}")

        return metrics

    def backtest(self, train_size=0.8):
        """Performs rolling-window backtesting."""
        logger.info("Starting backtest...")
        split_idx = int(len(self.returns) * train_size)
        train, test = self.returns.iloc[:split_idx], self.returns.iloc[split_idx:]

        backtest_predictions = []
        actual_volatility = test.var(axis=1)

        for i in range(len(test)):
            # Rebuild the model for each window
            if self.model_type == "DCC":
                model = pf.DCCGARCH(train, p=self.p, q=self.q)
            elif self.model_type == "BEKK":
                model = pf.BEKKGARCH(train, p=self.p, q=self.q)

            fitted_model = model.fit()
            forecast = fitted_model.predict(h=1)
            backtest_predictions.append(forecast.iloc[0])

            # Update the training set
            train = pd.concat([train, test.iloc[[i]]])

        predicted_volatility = pd.DataFrame(backtest_predictions, index=test.index, columns=["Predicted Volatility"])
        actual_volatility = pd.DataFrame(actual_volatility, index=test.index, columns=["Actual Volatility"])

        mse = mean_squared_error(actual_volatility, predicted_volatility)
        logger.info(f"Backtesting MSE: {mse}")

        return {"MSE": mse}

    def plot_forecast(self, steps=30):
        """Plots predicted volatility over time."""
        forecast = self.forecast(steps)

        # Ensure the index is a datetime index
        if not isinstance(self.data.index, pd.DatetimeIndex):
            self.data.index = pd.to_datetime(self.data.index)

        plt.figure(figsize=(12, 6))

        for target in self.target_cols:
            plt.plot(forecast.index, forecast[target], label=f"Predicted {target} Volatility")

        plt.legend()
        plt.title(f"MGARCH {self.model_type} Forecast")
        plt.xlabel("Time")
        plt.ylabel("Volatility")
        plt.show()


# import numpy as np
# import pandas as pd
# from arch import arch_model
# from mgarch.mgarch import DCCMGARCH
# from sklearn.preprocessing import StandardScaler

# class MGARCHModel:
#     def __init__(self, data_path):
#         self.data_path = data_path
#         self.data = None
#         self.model = None
#         self.returns = None

#     def load_data(self):
#         """Loads Brent oil prices and economic indicators."""
#         df = pd.read_csv(self.data_path, parse_dates=["Date"])
#         df.set_index("Date", inplace=True)
#         df.dropna(inplace=True)

#         # Compute log returns for all series
#         df_returns = df.pct_change().dropna()

#         # Standardize returns for numerical stability
#         scaler = StandardScaler()
#         self.returns = pd.DataFrame(scaler.fit_transform(df_returns), index=df_returns.index, columns=df_returns.columns)

#         self.data = df
#         return self.returns

#     def fit_model(self):
#         """Fits a DCC-GARCH model."""
#         self.model = DCCMGARCH(p=1, q=1)
#         self.model.fit(self.returns)
#         return self.model

#     def forecast_volatility(self, steps=5):
#         """Forecasts conditional volatility and correlations."""
#         vol_forecast, corr_forecast = self.model.forecast(steps)
#         return vol_forecast, corr_forecast

# import numpy as np
# import pandas as pd
# from sklearn.metrics import mean_squared_error
# from mgarch_model import MGARCHModel

# class MGARCHBacktester:
#     def __init__(self, data_path, train_ratio=0.8):
#         self.data_path = data_path
#         self.train_ratio = train_ratio
#         self.mgarch_model = MGARCHModel(data_path)
#         self.train_data = None
#         self.test_data = None
#         self.model = None

#     def split_data(self):
#         """Splits data into training and testing sets."""
#         returns = self.mgarch_model.load_data()
#         split_idx = int(len(returns) * self.train_ratio)
#         self.train_data = returns.iloc[:split_idx]
#         self.test_data = returns.iloc[split_idx:]
#         return self.train_data, self.test_data

#     def train_model(self):
#         """Trains MGARCH model on training data."""
#         self.mgarch_model.returns = self.train_data
#         self.model = self.mgarch_model.fit_model()
#         return self.model

#     def evaluate_forecast(self):
#         """Evaluates forecast accuracy on test data."""
#         actual_volatility = self.test_data.std(axis=0).values
#         vol_forecast, _ = self.mgarch_model.forecast_volatility(steps=len(self.test_data))

#         # Compute RMSE and MSE for volatility predictions
#         rmse = np.sqrt(mean_squared_error(actual_volatility, vol_forecast.mean(axis=0)))
#         mse = mean_squared_error(actual_volatility, vol_forecast.mean(axis=0))

#         return {"RMSE": rmse, "MSE": mse}

# import itertools
# import numpy as np
# import pandas as pd
# from arch import arch_model
# from mgarch_model import MGARCHModel
# from sklearn.metrics import mean_squared_error

# class MGARCHTuner:
#     def __init__(self, data_path, p_values, q_values, distributions):
#         self.data_path = data_path
#         self.p_values = p_values
#         self.q_values = q_values
#         self.distributions = distributions
#         self.best_params = None
#         self.best_score = float("inf")

#     def tune_hyperparameters(self):
#         """Grid search over (p, q) and distribution assumptions."""
#         mgarch = MGARCHModel(self.data_path)
#         returns = mgarch.load_data()

#         # Split data
#         train_size = int(len(returns) * 0.8)
#         train_data = returns.iloc[:train_size]
#         test_data = returns.iloc[train_size:]

#         # Iterate over all combinations of p, q, and distributions
#         for p, q, dist in itertools.product(self.p_values, self.q_values, self.distributions):
#             try:
#                 # Fit model
#                 mgarch.returns = train_data
#                 model = mgarch.fit_model(p=p, q=q, dist=dist)

#                 # Forecast test set volatility
#                 vol_forecast, _ = mgarch.forecast_volatility(steps=len(test_data))

#                 # Compute RMSE
#                 actual_volatility = test_data.std(axis=0).values
#                 rmse = np.sqrt(mean_squared_error(actual_volatility, vol_forecast.mean(axis=0)))

#                 # Update best parameters
#                 if rmse < self.best_score:
#                     self.best_score = rmse
#                     self.best_params = (p, q, dist)
                
#                 print(f"Tried (p={p}, q={q}, dist={dist}) -> RMSE: {rmse:.4f}")

#             except Exception as e:
#                 print(f"Failed for (p={p}, q={q}, dist={dist}): {e}")

#         print(f"Best Model: p={self.best_params[0]}, q={self.best_params[1]}, dist={self.best_params[2]}, RMSE={self.best_score:.4f}")
#         return self.best_params

