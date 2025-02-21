import os, sys
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt

# Setup logger for visualizer
sys.path.append(os.path.join(os.path.abspath(__file__), '..', '..', '..'))
from scripts.utils.logger import setup_logger

logger = setup_logger("visualizer")

class BrentOilVisualizer:
    def __init__(self, data_path: str, plot_dir: str = None):
        """Initialize EDA with cleaned Brent oil dataset."""
        self.data_path = data_path
        self.data = None
        self.plot_dir = Path(plot_dir) if plot_dir else Path("../screenshots/plots")
        self.plot_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self):
        """Load preprocessed Brent oil price dataset."""
        self.data = pd.read_csv(self.data_path, parse_dates=['Date'])
        logger.info(f"Data loaded successfully. Shape: {self.data.shape}")

    def summarize_data(self):
        """Display summary statistics and missing values."""
        logger.info("\nDataset Summary:")
        logger.info(self.data.info())
        logger.info("\nDescriptive Statistics:")
        logger.info(self.data.describe())
        logger.info("\nMissing Values:")
        logger.info(self.data.isnull().sum())

    def plot_trend(self):
        """Plot Brent oil price trend over time."""
        plt.figure(figsize=(14, 6))
        plt.plot(self.data['Date'], self.data['Price'], label='Brent Oil Price', color='blue')
        plt.xlabel("Year")
        plt.ylabel("Price (USD per Barrel)")
        plt.title("Brent Oil Price Trend")
        plt.legend()
        plt.savefig(self.plot_dir / "brent_oil_price_trend.png")
        plt.show()

    def plot_seasonality(self):
        """Visualize seasonal trends using yearly averages."""
        self.data['Year_Month'] = self.data['Date'].dt.to_period('M')
        monthly_avg = self.data.groupby('Year_Month')['Price'].mean()

        plt.figure(figsize=(14, 6))
        monthly_avg.plot()
        plt.xlabel("Year")
        plt.ylabel("Average Price (USD per Barrel)")
        plt.title("Monthly Average Brent Oil Price Trend")
        plt.savefig(self.plot_dir / "monthly_average_brent_oil_price_trend.png")
        plt.show()

    def plot_volatility(self):
        """Analyze price volatility using rolling standard deviation."""
        self.data['Rolling_STD'] = self.data['Price'].rolling(window=365).std()

        plt.figure(figsize=(14, 6))
        plt.plot(self.data['Date'], self.data['Rolling_STD'], color='red', label="Rolling Std Dev (365 days)")
        plt.xlabel("Year")
        plt.ylabel("Volatility")
        plt.title("Brent Oil Price Volatility")
        plt.legend()
        plt.savefig(self.plot_dir / "brent_oil_price_volatility.png")
        plt.show()

    def plot_correlation(self):
        """Compute and visualize correlation between price and time-based features."""
        corr_matrix = self.data[['Price', 'Year', 'Month', 'Weekday', 'Quarter']].corr()

        plt.figure(figsize=(8, 5))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.savefig(self.plot_dir / "correlation_heatmap.png")
        plt.show()

    def run_eda(self):
        """Execute the full EDA pipeline."""
        self.load_data()
        self.summarize_data()
        self.plot_trend()
        self.plot_seasonality()
        self.plot_volatility()
        self.plot_correlation()
