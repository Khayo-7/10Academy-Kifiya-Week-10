import os, sys
import numpy as np
import pandas as pd
from datetime import datetime

# Setup logger for cleaning
sys.path.append(os.path.join(os.path.abspath(__file__), '..', '..', '..'))
from scripts.utils.logger import setup_logger

logger = setup_logger("cleaning")

class BrentOilDataPreprocessor:
    def __init__(self, file_path: str):
        """Initialize the data preprocessor with the file path."""
        self.file_path = file_path
        self.data = None

    def load_data(self):
        """Load the Brent oil price dataset."""
        self.data = pd.read_csv(self.file_path)
        logger.info(f"Data loaded successfully. Shape: {self.data.shape}")
        return self
    
    def save_data(self, output_path):
        """Saves the cleaned and processed dataset."""
        self.data.to_csv(output_path, index=False)
    
    def get_data(self):
        """Returns the processed dataframe."""
        return self.data

    def clean_data(self):
        """Handle missing values and filter invalid price data."""
        self.data.dropna(subset=['Price'], inplace=True)
        self.data['Price'] = pd.to_numeric(self.data['Price'], errors='coerce')
        self.data.dropna(subset=['Price'], inplace=True)
        self.data = self.data[self.data['Price'] > 0]  # Remove negative/zero prices
        logger.info("Data cleaning completed.")

    def parse_dates(self):
        """Parses date column into a standardized format."""
        
        if 'Date' in self.data.columns:
            try:
                self.data["Date"] = pd.to_datetime(self.data["Date"], errors='coerce', dayfirst=True)
                # self.data["Date"] = pd.to_datetime(self.data["Date"], errors='coerce', format='%d/%m/%Y')
                self.data.dropna(subset=["Date"], inplace=True)
                self.data.sort_values(by="Date", inplace=True)
            except Exception as e:
                logger.error(f"Date parsing failed: {e}")
        else:
            logger.warning("Date column not found in data.")

        logger.info("Date parsing completed.")
        return self
    
    def handle_missing_values(self):
        """Handles missing values by forward-filling and backward-filling."""
        if 'Price' in self.data.columns:
            self.data["Price"] = pd.to_numeric(self.data["Price"], errors='coerce')
            self.data["Price"] = self.data["Price"].fillna(method='ffill')
            self.data["Price"] = self.data["Price"].fillna(method='bfill')
        else:
            logger.warning("Price column not found in data.")
        
        logger.info("Handled missing values.")
        return self
    
    def remove_outliers(self, method='zscore', threshold=3):
        """Removes outliers using Z-score or IQR method."""
        if method == 'zscore':
            z_scores = np.abs((self.data['Price'] - self.data['Price'].mean()) / self.data['Price'].std())
            self.data = self.data[z_scores < threshold]
        elif method == 'iqr':
            q1 = self.data['Price'].quantile(0.25)
            q3 = self.data['Price'].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            self.data = self.data[(self.data['Price'] >= lower_bound) & (self.data['Price'] <= upper_bound)]
        return self
    
    def feature_engineering(self):
        """Adds useful features like rolling averages."""
        self.data["7_day_MA"] = self.data["Price"].rolling(window=7, min_periods=1).mean()
        self.data["30_day_MA"] = self.data["Price"].rolling(window=30, min_periods=1).mean()
        self.data["Volatility"] = self.data["Price"].pct_change().rolling(window=30, min_periods=1).std()
        self.data.dropna(inplace=True)  # Remove NaNs from rolling calculations
        return self

    def add_time_features(self):
        """Generate additional time-based features for analysis."""
        self.data['year'] = self.data['Date'].dt.year
        self.data['month'] = self.data['Date'].dt.month
        self.data['day'] = self.data['Date'].dt.day
        self.data['weekday'] = self.data['Date'].dt.weekday
        self.data['quarter'] = self.data['Date'].dt.quarter
        logger.info("Time-based features added.")

    def preprocess(self):
        """Execute the full preprocessing pipeline."""
        self.load_data()
        self.parse_dates()
        self.clean_data()
        self.handle_missing_values()
        self.remove_outliers()
        self.add_time_features()
        self.feature_engineering()
        logger.info(f"Preprocessing completed. Final shape: {self.data.shape}")
        return self.get_data()
