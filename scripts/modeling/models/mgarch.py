import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch.univariate import ConstantMean, StudentsT
from arch.multivariate import DCC, BEKK
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from scripts.utils.logger import setup_logger

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
            mean_model = ConstantMean(self.returns)
            self.model = DCC(mean_model, p=self.p, q=self.q, dist=StudentsT())
        elif self.model_type == "BEKK":
            logger.info("Building BEKK-MGARCH model...")
            mean_model = ConstantMean(self.returns)
            self.model = BEKK(mean_model, p=self.p, q=self.q, dist=StudentsT())
        else:
            logger.error("Invalid model type. Choose 'DCC' or 'BEKK'.")
            raise ValueError("Invalid MGARCH model type.")

    def train(self):
        """Trains the MGARCH model."""
        if self.model is None:
            logger.error("Model not built. Run `build_model()` first.")
            raise ValueError("Run `build_model()` before training.")

        logger.info("Training MGARCH model...")
        self.fitted_model = self.model.fit(disp="off")
        logger.info(f"Model training completed. AIC: {self.fitted_model.aic}, BIC: {self.fitted_model.bic}")

    def forecast(self, steps=30):
        """Generates future volatility forecasts."""
        if self.fitted_model is None:
            logger.error("Model not trained. Run `train()` first.")
            raise ValueError("Run `train()` before forecasting.")

        logger.info(f"Forecasting {steps} steps ahead...")
        forecasts = self.fitted_model.forecast(horizon=steps)
        vol_forecast = forecasts.variance.iloc[-1]  # Extract the last forecasted variance
        return vol_forecast

    def evaluate(self):
        """Evaluates the MGARCH model."""
        if self.fitted_model is None:
            logger.error("Model not trained. Run `train()` first.")
            raise ValueError("Run `train()` before evaluation.")

        log_likelihood = self.fitted_model.loglikelihood
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
                model = DCC(ConstantMean(train), p=self.p, q=self.q, dist=StudentsT())
            elif self.model_type == "BEKK":
                model = BEKK(ConstantMean(train), p=self.p, q=self.q, dist=StudentsT())

            fitted_model = model.fit(disp="off")
            forecast = fitted_model.forecast(horizon=1).variance.iloc[-1]
            backtest_predictions.append(forecast)

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