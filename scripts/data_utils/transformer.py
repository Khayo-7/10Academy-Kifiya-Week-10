import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from statsmodels.tsa.stattools import adfuller

# Setup logger for transformer
sys.path.append(os.path.join(os.path.abspath(__file__), '..', '..', '..'))
from scripts.utils.logger import setup_logger

logger = setup_logger("transformer")

class BrentOilDataTransformer:
    def __init__(self, file_path: str):
        """Initialize the data transfomrer with the file path."""
        self.file_path = file_path
        self.data = None

    def load_data(self):
        """Load the Brent oil price dataset."""
        self.data = pd.read_csv(self.file_path, parse_dates=["Date"], dayfirst=True)
        self.data.sort_values(by="Date", inplace=True)
        self.data.set_index("Date", inplace=True)
        logger.info(f"Data loaded successfully. Shape: {self.data.shape}")
        return self
    
    def save_data(self, output_path):
        """Saves the transformed dataset."""
        self.data.reset_index(inplace=True)
        self.data.to_csv(Path(output_path), index=False)
    
    def get_data(self):
        """Returns the transformed dataframe."""
        return self.data

    def test_stationarity(self, column='Price', significance=0.05):
        """Perform Augmented Dickey-Fuller (ADF) test for stationarity."""
        logger.debug(f"Performing ADF test on column: {column}")
        result = adfuller(self.data[column].dropna())  # Drop NaN values for differencing
        adata_stat, p_value = result[0], result[1]
        is_stationary = p_value < significance
        
        logger.info("\nAugmented Dickey-Fuller Test Results:")
        logger.debug(f"ADF Statistic: {adata_stat}, p-value: {p_value}, Stationary: {is_stationary}")

        return {
            'ADF Statistic': adata_stat,
            'p-value': p_value,
            'Stationary': is_stationary
        }

    def apply_differencing(self, column='Price', diff_order=1):
        """Apply differencing to make the time series stationary. """
        logger.debug(f"Applying differencing on column: {column} with order: {diff_order}")
        diff_column = f"{column}_diff_{diff_order}"
        self.data[diff_column] = self.data[column].diff(periods=diff_order)
        logger.debug(f"Differenced column created: {diff_column}")
        return self

    def apply_log_transformation(self, column='Price'):
        """Apply log transformation to stabilize variance."""
        logger.debug(f"Applying log transformation on column: {column}")
        log_column = f"Log_{column}"
        self.data[log_column] = np.log(self.data[column])
        logger.debug(f"Log transformed column created: {log_column}")
        return self

    def apply_seasonal_differencing(self, column='Price', seasonality=365):
        """Apply seasonal differencing to remove seasonality."""
        logger.debug(f"Applying seasonal differencing on column: {column} with seasonality: {seasonality}")
        seasonal_diff_column = f"{column}_Seasonal_diff_{seasonality}"
        self.data[seasonal_diff_column] = self.data[column] - self.data[column].shift(seasonality)
        logger.debug(f"Seasonal differenced column created: {seasonal_diff_column}")
        return self

    def apply_transformations(self, column='Price', diff_order=1, seasonality=365):
        """Applies differencing, log transformation, and seasonal differencing."""
        self.apply_differencing(column, diff_order=diff_order)

        # Test stationarity after each transformation
        test_results = self.test_stationarity(f'{column}_diff_{diff_order}')
        
        if not test_results['Stationary']:
            self.apply_log_transformation(column)
            test_results = self.test_stationarity(f'Log_{column}')
        if not test_results['Stationary']:
            self.apply_seasonal_differencing(column, seasonality=seasonality)
            test_results = self.test_stationarity(f'{column}_Seasonal_diff_{seasonality}')
        
        # Drop NaNs caused by differencing
        logger.info("Applied transformations pipeline (differencing, log, seasonal differencing).")
        
        if not test_results['Stationary']:
            logger.debug(f"Transformation pipeline successfully made the time series stationary.")
        else:
            logger.debug(f"Transformation pipeline did not make the time series stationary.")

        self.data.dropna(inplace=True)
        return self

    def transform(self):
        """Execute the full transforming pipeline."""
        self.load_data()
        self.apply_transformations(column='Price', diff_order=1, seasonality=365)
        logger.info(f"Transformation completed. Final shape: {self.data.shape}")
        return self.get_data()
