import numpy as np
import pandas as pd
from mgarch_model import MGARCHModel
from lstm_model import LSTMPredictor
from sklearn.metrics import mean_squared_error

class ModelFinalizer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.results = {}

    def load_data(self):
        """Load and preprocess Brent oil price data."""
        df = pd.read_csv(self.data_path, parse_dates=['Date'], index_col='Date')
        df['Returns'] = df['Price'].pct_change().dropna()
        return df[['Returns']].dropna()

    def evaluate_model(self, predictions, actuals, model_name):
        """Compute RMSE and store results."""
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        self.results[model_name] = rmse
        print(f"{model_name} RMSE: {rmse:.4f}")

    def backtest_models(self):
        """Train and backtest MGARCH and LSTM models."""
        data = self.load_data()
        train_size = int(len(data) * 0.8)
        train, test = data.iloc[:train_size], data.iloc[train_size:]

        # 1. MGARCH Model
        mgarch = MGARCHModel(self.data_path)
        mgarch.fit_model(p=1, q=1, dist="t")
        mgarch_forecast, _ = mgarch.forecast_volatility(steps=len(test))
        self.evaluate_model(mgarch_forecast.mean(axis=0), test["Returns"], "MGARCH")

        # 2. LSTM Model
        lstm_model = LSTMPredictor(self.data_path)
        lstm_model.prepare_data(seq_length=10)
        lstm_model.train_model(epochs=10, batch_size=16)
        lstm_pred = lstm_model.forecast(steps=len(test))
        self.evaluate_model(lstm_pred, test["Returns"], "LSTM")

        print("\nFinal Model Selection:", min(self.results, key=self.results.get))

# Run Finalization
if __name__ == "__main__":
    finalizer = ModelFinalizer("brent_oil_prices_economic_indicators.csv")
    finalizer.backtest_models()
