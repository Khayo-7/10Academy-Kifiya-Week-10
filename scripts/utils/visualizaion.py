import os, sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Setup logger for visualization
sys.path.append(os.path.join(os.path.abspath(__file__), '..', '..', '..'))
from scripts.utils.logger import setup_logger

logger = setup_logger("visualization")

def plot_hist(data, title, figsize=(12, 8)):
    data.hist(figsize=figsize, bins=30)
    plt.suptitle(title)
    plt.show()

def plot_numerical(data, columns, figsize=(12, 6)):
        
    plt.figure(figsize=figsize)
    for i, column in enumerate(columns, 1):
        plt.subplot(2, 2, i)
        sns.histplot(data[column], bins=30, kde=True)
        plt.title(f"Distribution of {column}")
    plt.tight_layout()
    plt.show()

def plot_categorical(data, columns, palette="coolwarm", figsize=(12, 4)):
        
    plt.figure(figsize=figsize)
    for i, column in enumerate(columns, 1):
        plt.subplot(1, 3, i)
        sns.countplot(x=data[column], hue=data[column], palette=palette, legend=False)
        plt.title(f"Distribution of {column}")
    plt.tight_layout()

def plot_count(data, column, title, palette="Set2", hue=None, figsize=(6, 4)):

    plt.figure(figsize=figsize)
    sns.countplot(x=column, data=data, hue=hue if hue else column,  order=data[column].value_counts().index, 
                    palette=palette, legend=False)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.show()

def plot_box(data, x, y, title, palette="Set2", hue=None, figsize=(6, 4)):
    plt.figure(figsize=figsize)
    sns.boxplot(x=x, y=y, data=data, hue=hue if hue else x, palette=palette, legend=False)
    plt.title(title)
    plt.xticks()
    plt.show()

def plot_correlation(data, title, figsize=(10, 6)):
    plt.figure(figsize=figsize)
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title(title)
    plt.show()

def plot_correlation(data: pd.DataFrame, title: str, figsize=(10, 6)):

    numeric_data = data.select_dtypes(include=['int64', 'float64']) # Select only numeric columns
    plt.figure(figsize=figsize)
    sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title(title)
    plt.show()

def plot_bar(data, title, palette="viridis", figsize=(10, 6)):
    plt.figure(figsize=figsize)
    sns.barplot(x=data.index, y=data.values, palette=palette)
    plt.xticks(rotation=45)
    plt.title(title)
    plt.show()

def plot_line(data, x, y, title, xlabel, ylabel, figsize=(10, 5)):
    plt.figure(figsize=figsize)
    sns.lineplot(x=x, y=y, data=data, estimator=sum, errorbar=None)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def plot_feature_importance(model, X, title=""):
    importances = pd.Series(model.feature_importances_, index=X.columns)
    importances.sort_values(ascending=False).plot(kind="bar", figsize=(12, 5), colormap="viridis")
    plt.title(title)
    plt.show()