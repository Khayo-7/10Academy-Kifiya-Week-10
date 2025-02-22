import os, sys, ipaddress
import numpy as np
import pandas as pd
from io import StringIO

# Setup logger for utils
sys.path.append(os.path.join(os.path.abspath(__file__), '..', '..', '..'))
from scripts.utils.logger import setup_logger

logger = setup_logger("utils")

def summarize_data(data: pd.DataFrame):
    """
    Prints summary statistics and detailed information about the dataset, including duplications.
    """
    try:
        logger.info("\n--- Data Summary ---")
        summary_stats = data.describe(include="all").transpose()
        summary_stats['unique'] = data.nunique()
        summary_stats['dtype'] = data.dtypes
        logger.info(summary_stats)
        
        logger.info("\n--- Data Info ---")
        buffer = StringIO() # Use StringIO to capture output
        data.info(buf=buffer)
        logger.info(buffer.getvalue())
        
        logger.info("\n--- Missing Values Analysis ---")
        missing_summary = data.isnull().sum().reset_index()
        missing_summary.columns = ['Column', 'MissingCount']
        missing_summary['MissingPercentage'] = (missing_summary['MissingCount'] / len(data)) * 100
        missing_summary = missing_summary[missing_summary['MissingCount'] > 0]
        
        if not missing_summary.empty:
            logger.info("Columns with missing values:")
            logger.info(missing_summary)
        else:
            logger.info("No missing values found in the dataset.")
        
        logger.info("\n--- Total columns with Missing Values ---")
        logger.info((data.isnull().sum() > 0).sum())
        
        # Check for duplicate rows
        logger.info("\n--- Duplication Analysis ---")
        duplicate_rows = data.duplicated().sum()
        if duplicate_rows > 0:
            logger.info(f"Number of duplicate rows: {duplicate_rows}")
            logger.info("Sample of duplicate rows:")
            logger.info(data[data.duplicated(keep=False)].head())  # Show a sample of duplicate rows
        else:
            logger.info("No duplicate rows found.")
        
        # # Check for duplicate columns
        # duplicate_columns = data.T.duplicated().sum()
        # if duplicate_columns > 0:
        #     logger.info(f"Number of duplicate columns: {duplicate_columns}")
        #     logger.info("Duplicate columns:")
        #     logger.info(data.columns[data.T.duplicated()].tolist())
        # else:
        #     logger.info("No duplicate columns found.")
        
        logger.info("\n--- Data Skewness ---")
        logger.info(data.skew(numeric_only=True))
        
        logger.info("\n--- Data Kurtosis ---")
        logger.info(data.kurtosis(numeric_only=True))
        
        logger.info("\n--- Correlation Matrix ---")
        logger.info(data.corr(numeric_only=True))
        
        logger.info("\n--- Data Head ---")
        logger.info(data.head())
        
        logger.info("\n--- Data Tail ---")
        logger.info(data.tail())
        
        logger.info("Data summary and analysis completed successfully.")
        
    except Exception as e:
        logger.error(f"Error summarizing data: {e}", exc_info=True)
        raise
