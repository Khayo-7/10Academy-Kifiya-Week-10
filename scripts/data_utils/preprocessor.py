import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Setup logger for preprocessor
sys.path.append(os.path.join(os.path.abspath(__file__), '..', '..', '..'))
from scripts.utils.logger import setup_logger

logger = setup_logger("preprocessor")

class BrentOilDataPreprocessor:
    def __init__(self, file_path: str):
        """Initialize the data preprocessor with the file path."""
        self.file_path = file_path
        self.data = None

    def load_data(self):
        """Load the Brent oil price dataset."""
        self.data = pd.read_csv(self.file_path, parse_dates=["Date"], dayfirst=True) # index_col="Date")
        self.data.sort_values(by="Date", inplace=True)
        self.data.set_index("Date", inplace=True)
        logger.info(f"Data loaded successfully. Shape: {self.data.shape}")
        return self
    
    def save_data(self, output_path):
        """Saves the cleaned and processed dataset."""
        self.data.reset_index(inplace=True)
        self.data.to_csv(Path(output_path), index=False)
    
    def get_data(self):
        """Returns the processed dataframe."""
        return self.data

    def clean_data(self, column='Price'):
        """Handle missing values and filter invalid price data."""
        self.data.dropna(subset=[column], inplace=True)
        self.data[column] = pd.to_numeric(self.data[column], errors='coerce')
        self.data.dropna(subset=[column], inplace=True)
        self.data = self.data[self.data[column] > 0]  # Remove negative/zero prices
        logger.info("Data cleaning completed.")
        return self

    def parse_dates(self):
        """Parses date column into a standardized format and sets it as index."""
        if 'Date' in self.data.columns:
            try:
                self.data["Date"] = pd.to_datetime(self.data["Date"], errors='coerce', dayfirst=True) # , format='%d/%m/%Y')
                self.data.dropna(subset=["Date"], inplace=True)
                self.data.sort_values(by="Date", inplace=True)
                self.data.set_index("Date", inplace=True)
            except Exception as e:
                logger.error(f"Date parsing failed: {e}")
        else:
            logger.warning("Date column not found in data.")

        logger.info("Date parsing completed.")
        return self
    
    def handle_missing_values(self, column='Price'):
        """Handles missing values by forward-filling and backward-filling."""
        if column in self.data.columns:
            self.data[column] = pd.to_numeric(self.data[column], errors='coerce')
            self.data[column] = self.data[column].fillna(method='ffill')
            self.data[column] = self.data[column].fillna(method='bfill')
        else:
            logger.warning(f"{column} column not found in data.")
        
        logger.info("Handled missing values.")
        return self
    
    def remove_outliers(self, column='Price', method='zscore', threshold=3, apply_removal=True):
        """Removes outliers using Z-score or IQR method. Set apply_removal=False to skip."""
        if apply_removal:
            if method == 'zscore':
                z_scores = np.abs((self.data[column] - self.data[column].mean()) / self.data[column].std())
                self.data = self.data[z_scores < threshold]
            elif method == 'iqr':
                q1 = self.data[column].quantile(0.25)
                q3 = self.data[column].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                self.data = self.data[(self.data[column] >= lower_bound) & (self.data[column] <= upper_bound)]
            logger.info(f"Outliers removed using {method} method.")
        else:
            logger.info("Outlier removal skipped.")

        return self
    
    def generate_temporal_features(self):
        """Generate additional time-based features for analysis."""
        self.data['Year'] = self.data.index.year
        # self.data['Year'] = self.data['Date'].dt.year
        self.data['Month'] = self.data.index.month
        # self.data['Month'] = self.data['Date'].dt.month
        self.data['Day'] = self.data.index.day
        # self.data['Day'] = self.data['Date'].dt.day
        self.data['Weekday'] = self.data.index.weekday
        # self.data['Weekday'] = self.data['Date'].dt.weekday
        self.data['Quarter'] = self.data.index.quarter
        # self.data['Quarter'] = self.data['Date'].dt.quarter
        self.data['Is_Weekend'] = self.data['Weekday'].apply(lambda x: 1 if x > 4 else 0)
        self.data['Year_Month'] = self.data.index.to_period('M')
        # self.data['Year_Month'] = self.data['Date'].dt.to_period('M')
        logger.info("Time-based features added.")
        return self
        
    def feature_engineering(self, column='Price'):
        """Adds useful features like rolling averages and volatility."""
        self.data["7_day_MA"] = self.data[column].rolling(window=7, min_periods=1).mean()
        self.data["30_day_MA"] = self.data[column].rolling(window=30, min_periods=1).mean()
        self.data['Rolling_STD'] = self.data[column].rolling(window=365, min_periods=1).std()
        self.data["Volatility"] = self.data[column].pct_change().rolling(window=30, min_periods=1).std()
        self.data.dropna(inplace=True) # Remove NaNs from rolling calculations
        logger.info("Feature engineering completed.")
        return self

    def preprocess(self, remove_outliers=True):
        """Execute the full preprocessing pipeline."""
        self.load_data()
        self.parse_dates()
        self.clean_data()
        self.handle_missing_values()
        self.remove_outliers(apply_removal=remove_outliers)
        self.generate_temporal_features()
        self.feature_engineering()
        logger.info(f"Preprocessing completed. Final shape: {self.data.shape}")
        return self.get_data()

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import ruptures as rpt
# import pymc3 as pm

# data['Date'] = pd.to_datetime(data['Date'])
# data = data.sort_values(by='Date')
# data = data.set_index('Date')

# # Preprocess the data
# data['Price'] = data['Price'].interpolate()
# # Plot the data
# plt.figure(figsize=(14, 7))
# plt.plot(data.index, data['Price'], label='Brent Oil Price')
# plt.xlabel('Date')
# plt.ylabel('Price (USD)')
# plt.title('Brent Oil Prices Over Time')
# plt.legend()
# plt.show()

# # CUSUM method
# mean_price = data['Price'].mean()
# cusum = np.cumsum(data['Price'] - mean_price)
# plt.figure(figsize=(14, 7))
# plt.plot(data.index, cusum, label='CUSUM')
# plt.axhline(y=0, color='r', linestyle='--')
# plt.xlabel('Date')
# plt.ylabel('CUSUM Value')
# plt.title('CUSUM Analysis')
# plt.legend()
# plt.show()

# # Bayesian Change Point Detection using PyMC3
# with pm.Model() as model:
#     # Priors
#     mean_prior = pm.Normal('mean_prior', mu=mean_price, sigma=10)
#     change_point = pm.DiscreteUniform('change_point', lower=0, upper=len(data)-1)

#     # Likelihood
#     likelihood = pm.Normal('likelihood', mu=mean_prior, sigma=10, observed=data['Price'])

#     # Inference
#     trace = pm.sample(1000, tune=1000, cores=2)

# # Plot results
# pm.plot_posterior(trace)
# plt.show()


# change_point_index = 4753

# # Plot the data with the change point
# plt.figure(figsize=(14, 7))
# plt.plot(data.index, data['Price'], label='Brent Oil Price')
# plt.axvline(x=data.index[change_point_index], color='red', linestyle='--', label='Change Point')
# plt.xlabel('Date')
# plt.ylabel('Price (USD)')
# plt.title('Brent Oil Prices with Change Point')
# plt.legend()
# plt.show()


# price_array = data['Price'].values
# model = "rbf"
# algo = rpt.Pelt(model=model).fit(price_array)
# change_points = algo.predict(pen=20)

# plt.figure(figsize=(14, 7))
# plt.plot(data.index, data['Price'], label='Brent Oil Price')
# for cp in change_points[:-1]:
#     plt.axvline(x=data.index[cp], color='red', linestyle='--', label='Change Point' if cp == change_points[0] else "")
# plt.xlabel('Date')
# plt.ylabel('Price (USD)')
# plt.title('Brent Oil Prices with Detected Change Points')
# plt.legend()
# plt.show()