import os, sys
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from statsmodels.tsa.stattools import adfuller

# Setup logger for data_loader
sys.path.append(os.path.join(os.path.abspath(__file__), '..', '..', '..'))
from scripts.utils.logger import setup_logger

logger = setup_logger("stationarity")

def test_stationarity(data, column='Price', significance=0.05):
    """Perform Augmented Dickey-Fuller (ADF) test for stationarity."""
    logger.debug(f"Performing ADF test on column: {column}")
    result = adfuller(data[column].dropna())  # Drop NaN values for differencing
    adata_stat, p_value = result[0], result[1]
    is_stationary = p_value < significance
    
    logger.info("\nAugmented Dickey-Fuller Test Results:")
    logger.debug(f"ADF Statistic: {adata_stat}, p-value: {p_value}, Stationary: {is_stationary}")

    return {
        'ADF Statistic': adata_stat,
        'p-value': p_value,
        'Stationary': is_stationary
    }

def apply_differencing(data, column='Price', diff_order=1):
    """Apply differencing to make the time series stationary. """
    logger.debug(f"Applying differencing on column: {column} with order: {diff_order}")
    diff_column = f"{column}_diff_{diff_order}"
    data[diff_column] = data[column].diff(periods=diff_order)
    logger.debug(f"Differenced column created: {diff_column}")
    return data

def apply_log_transformation(data, column='Price'):
    """Apply log transformation to stabilize variance."""
    logger.debug(f"Applying log transformation on column: {column}")
    log_column = f"Log_{column}"
    data[log_column] = np.log(data[column])
    logger.debug(f"Log transformed column created: {log_column}")
    return data

def apply_seasonal_differencing(data, column='Price', seasonality=365):
    """Apply seasonal differencing to remove seasonality."""
    logger.debug(f"Applying seasonal differencing on column: {column} with seasonality: {seasonality}")
    seasonal_diff_column = f"{column}_Seasonal_diff_{seasonality}"
    data[seasonal_diff_column] = data[column] - data[column].shift(seasonality)
    logger.debug(f"Seasonal differenced column created: {seasonal_diff_column}")
    return data

def stationary_pipeline(data, column='Price', diff_order=1, seasonality=365):
    """Apply a series of transformations to make the time series stationary."""
    logger.debug(f"Applying transformation pipeline on column: {column}")
    data = apply_differencing(data, column, diff_order)
    # data = apply_log_transformation(data, column)
    # data = apply_seasonal_differencing(data, column, seasonality)

    # Test stationarity after each transformation
    test_results = test_stationarity(data, 'Price_diff_1')
    
    if not test_results['Stationary']:
        data = apply_log_transformation(data, column)
        test_results = test_stationarity(data, 'Log_Price')
    if not test_results['Stationary']:
        data = apply_seasonal_differencing(data, column, seasonality)
        test_results = test_stationarity(data, 'Price_Seasonal_diff_365')
    
    logger.debug(f"Transformation pipeline applied successfully.")
    
    if not test_results['Stationary']:
        logger.debug(f"Transformation pipeline successfully made the time series stationary.")
    else:
        logger.debug(f"Transformation pipeline did not make the time series stationary.")

    return data, test_results