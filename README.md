# 10Academy-Kifiya-Week-10

# **Brent Oil Price Volatility Forecasting - The Impact of Key Global Events: From Analysis to Deployment**  

## **Introduction**

Understanding the volatility of Brent crude oil prices is essential for investors, policymakers, and energy companies to make informed decisions. Market fluctuations are often influenced by geopolitical events, economic policies, and supply-demand imbalances. To provide a data-driven approach for predicting oil price volatility, a robust time series modeling pipeline was developed, leveraging advanced statistical and deep learning techniques.  

Brent crude oil is a key benchmark for global oil pricing, influencing economies, financial markets, and energy policies worldwide. However, its price is highly volatile, often reacting sharply to political, economic, and geopolitical events. This analysis focuses on examining how major global events, such as **political decisions, regional conflicts, economic sanctions, and OPEC policy changes**, affect Brent oil prices. The goal is to provide data-driven insights that help stakeholders navigate the complexities of the energy market. This article walks through the key steps involved, from data exploration to deploying a predictive dashboard.  

---

## **Business Context**

Birhan Energies, a consultancy firm specializing in energy sector analytics, requires a systematic approach to studying **the impact of major global events on Brent oil prices**. The findings will support decision-making in the following areas:

- **Investment Strategies** – Helping investors assess risks and opportunities in the oil market.
- **Policy Development** – Assisting policymakers in designing strategies for energy security and economic stability.
- **Operational Planning** – Enabling energy companies to forecast price trends and optimize supply chains.

Given the unpredictable nature of the oil market, this analysis aims to quantify the influence of significant events on price fluctuations and improve forecasting accuracy.

---

## **Dataset Overview**

The dataset consists of **historical daily Brent oil prices** spanning from **May 20, 1987, to September 30, 2022**. It contains the following two primary fields:  

- **Date** – The daily timestamp of the recorded oil prices (formatted as `day-month-year`).
- **Price** – The Brent crude oil price on that date, recorded in **USD per barrel**.

This dataset serves as the foundation for analyzing price trends and evaluating the impact of external factors on oil price movements.

---

## **📌 Step 1: Data Analysis & Preprocessing**  

## **Task 1.1: Defining the Data Analysis Workflow**

Before conducting any predictive modeling, a structured workflow is necessary to ensure a systematic and reliable approach to the analysis.

### **1. Data Collection**

To assess the effects of external events on oil prices, additional datasets were identified, including:

- **Economic Indicators** – GDP growth, inflation, exchange rates, and unemployment rates from sources such as the **World Bank, IMF, and OECD**.
- **Geopolitical Events** – Key political events, conflicts, and trade sanctions from **news APIs and government reports**.
- **OPEC Policies** – Historical decisions on production quotas and supply adjustments.
- **Technological Advancements** – Trends in renewable energy and oil extraction technologies.

These datasets provide context for identifying relationships between oil prices and external shocks.

---

### **2. Data Preprocessing**

Given the long time span of the dataset, careful preprocessing was required to ensure data integrity and consistency. Steps included:

- **Handling Missing Data** – Addressing gaps in price data through interpolation techniques.
- **Time Alignment** – Standardizing different datasets to a common time frame (daily, monthly, or quarterly).
- **Normalization** – Scaling variables for better comparability in multivariate analyses.

Preprocessing ensures that the data is clean and ready for meaningful analysis. 

The data was cleaned, missing values were handled, and returns were calculated using:  

\[
R_t = \frac{P_t - P_{t-1}}{P_{t-1}}
\]

where \( P_t \) is the price at time \( t \).  
---

### **3. Exploratory Data Analysis (EDA)**

An exploratory analysis was performed to uncover initial patterns and relationships:

- **Trend Analysis** – Long-term price movements and periods of sustained increases or declines.
- **Seasonality Detection** – Recurring fluctuations in oil prices over time.
- **Volatility Measurement** – Identifying periods of high instability using rolling standard deviations.
- **Correlation Analysis** – Examining relationships between oil prices and macroeconomic indicators.

Visualizing these patterns provided key insights into factors influencing Brent oil price movements.

Key insights from exploratory data analysis:  

- **High volatility during economic crises**, including the 2008 financial collapse and the 2020 COVID-19 shock.  
- **Seasonal patterns and trends**, suggesting periodic price fluctuations.  
- **Non-stationarity**, requiring transformations before modeling.  

---
## **Task 1.2: Understanding the Models for Oil Price Analysis**

Selecting appropriate models is critical for accurately capturing the dynamics of oil price fluctuations. Several time series models were reviewed based on their ability to analyze price trends, volatility, and external influences.

### **1. ARIMA (AutoRegressive Integrated Moving Average)**
- **Purpose:** Forecasting trends and seasonality in oil prices.
- **Key Assumption:** Requires data to be stationary (constant mean and variance).

### **2. GARCH (Generalized Autoregressive Conditional Heteroskedasticity)**
- **Purpose:** Modeling price volatility and risk assessment.
- **Key Feature:** Captures periods of high and low fluctuations in oil prices.

### **3. VAR (Vector Autoregression)**
- **Purpose:** Examining relationships between oil prices and multiple economic indicators.
- **Key Feature:** Incorporates GDP, exchange rates, and inflation in a multivariate framework.

### **4. LSTM (Long Short-Term Memory Networks)**
- **Purpose:** Capturing complex, nonlinear dependencies in oil price movements.
- **Key Feature:** Effective for long-term sequential pattern recognition.

### **5. Markov-Switching ARIMA**
- **Purpose:** Identifying different market regimes (e.g., stable vs. volatile periods).
- **Key Feature:** Allows dynamic switching between different ARIMA models based on market conditions.

Each model addresses different aspects of oil price behavior, providing a comprehensive framework for analysis.

---

## **Key Takeaways from Task 1**

✔ A **structured data analysis workflow** was defined to guide the study.
✔ Key **datasets** were identified, including economic, political, and technological factors.
✔ **Data preprocessing** steps were implemented to clean and align the information.
✔ **Exploratory data analysis** provided initial insights into price trends, volatility, and correlations.
✔ Multiple **time series models** were reviewed for analyzing and forecasting oil prices.

---

## **📌 Step 2: Time Series Modeling for Volatility**  

To model oil price volatility, **GARCH and LSTM models** were implemented.  

### **1️⃣ GARCH Model (Generalized Autoregressive Conditional Heteroskedasticity)**  

GARCH was used to capture time-dependent volatility. Model selection involved tuning parameters \( (p, q) \) and choosing a suitable distribution (Gaussian vs. t-distribution).  

Best configuration:  
- **GARCH(1,1)** with **t-distribution**, as it better captured extreme fluctuations.  
- **Rolling forecasts** provided short-term volatility predictions.  

### **2️⃣ LSTM Model (Long Short-Term Memory Network)**  

To incorporate long-term dependencies in price movements, an LSTM model was trained. Key features:  
- **10-day sequence length** to learn patterns over time.  
- **Batch normalization and dropout layers** to prevent overfitting.  
- **Adam optimizer with MSE loss function** for efficient learning.  

Both models were backtested on unseen data, comparing their forecasting accuracy.  

---

## **📌 Step 3: Model Selection & Evaluation**  

Performance was measured using **Root Mean Squared Error (RMSE)** on test data.  

| Model   | RMSE  | Interpretation |
|---------|------:|---------------|
| GARCH(1,1)  | 0.0321 | Captures volatility well but struggles with long-term trends. |
| LSTM  | **0.0214** | Adapts better to price fluctuations, capturing both short-term and long-term dependencies. |

LSTM emerged as the final model due to its superior forecasting accuracy.  

---

## **📌 Step 4: Deployment via Flask & React**  

With the best model selected, deployment was the next step. A **Flask API** was developed to serve real-time price forecasts. The API accepts a date range as input and returns predicted prices.  

The frontend was built using **React.js**, integrating:  
- **Chart.js for visualizing predictions**  
- **Axios for API requests**  
- **User input fields for custom forecasts**  

---

## **📌 Final Outcome**  

The deployed **Brent Oil Price Forecast Dashboard** provides:  
✅ **Real-time volatility forecasts** using an LSTM model.  
✅ **Interactive visualization** of historical trends and predictions.  
✅ **Scalability for future enhancements**, including geopolitical event tracking.  

Understanding oil price fluctuations has never been more critical. A data-driven approach helps stakeholders make informed decisions in a volatile energy market.
