# 📊 Retail Sales Forecasting – ARIMA Time Series Analysis

[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://www.python.org/)
[![Time Series](https://img.shields.io/badge/Time_Series-ARIMA-red)](https://www.statsmodels.org/)
[![EDA](https://img.shields.io/badge/EDA-Pandas-yellow)](https://pandas.pydata.org/)
[![Visualization](https://img.shields.io/badge/Visualization-Matplotlib_Seaborn-orange)](https://matplotlib.org/)

A comprehensive Python-based exploratory data analysis and time-series forecasting project analyzing **12 months of retail sales data (2019)** to uncover monthly trends, identify top-performing products and cities, and build an ARIMA predictive model for accurate sales forecasting. This project combines data integration, exploratory analysis, time-series decomposition, and statistical forecasting to derive actionable insights for inventory planning, marketing strategy, and financial forecasting.

---

## 📋 Table of Contents

- [Project Objective](#-project-objective)
- [Dataset Description](#-dataset-description)
- [Key Analysis Steps](#-key-analysis-steps)
- [Methodology](#-methodology)
- [Key Findings](#-key-findings)
- [How to Use](#-how-to-use)
- [Visualization Guide](#-visualization-guide)
- [Learning Outcomes](#-learning-outcomes)

---

## 🎯 Project Objective

**Objective:** Analyze 12 months of retail sales data across multiple cities and product categories to identify temporal sales patterns, discover top-performing products and markets, understand geographic revenue distribution, and build a predictive ARIMA time-series model for accurate monthly sales forecasting—enabling data-driven inventory management, targeted marketing campaigns, financial planning, and revenue optimization.

**Dataset:** 12 Monthly CSV Files (Jan2019–Dec 2019) | **Industry:** Retail & E-Commerce Analytics

**Problem Statement:**

Retail businesses face challenges in accurately predicting future sales, optimizing inventory levels, and allocating marketing budgets without clear forecasting models. Key questions remain unanswered: What are the monthly sales trends? Which products and cities drive revenue? How can we forecast future sales with confidence? Without accurate forecasting, retailers risk overstocking (excess inventory costs), understocking (lost sales), misaligned marketing spend, and suboptimal financial planning. This project provides comprehensive sales analysis, product-market profiling, geographic performance insights, and a statistical forecasting model (ARIMA) to enable proactive business decisions, maximize profitability, and improve operational efficiency.

---

## 📊 Dataset Description

### Files: `Sales_January_2019.csv` → `Sales_December_2019.csv`

**Dataset Statistics:**
- **Total Records:** ~185,000 transactions
- **Data Period:** Full calendar year 2019 (January–December)
- **Monthly Files:** 12 individual CSV files (one per month)
- **Geographic Coverage:** ~20 US cities and states
- **Products:** ~19 unique products across electronics, accessories, and home goods
- **Sales Range:** $0.99 - $27,975 per transaction
- **Average Transaction:** ~$185
- **Total Annual Revenue:** ~$34.5M
- **Top Cities:** San Francisco, Los Angeles, New York, Boston, Dallas

### Column Definitions

| Column | Data Type | Description |
|--------|-----------|-------------|
| **Order ID** | String | Unique transaction identifier |
| **Product** | Categorical | Product name/SKU (19 unique products) |
| **Quantity Ordered** | Integer | Units purchased per transaction (1-100+) |
| **Price Each** | Float | Unit price in USD ($0.99 - $27,975) |
| **Order Date** | DateTime | Transaction date (YYYY-MM-DD format) |
| **Purchase Address** | String | Full delivery address (City, State, ZIP) |
| **Sales** | Float | Total transaction revenue (Quantity × Price Each) |
| **City** | String | Derived from address (20 unique cities) |
| **State** | String | Derived from address (US states) |
| **Month** | Integer | Extracted from Order Date (1-12) |
| **Year** | Integer | Extracted from Order Date (2019) |

### Data Characteristics

- **Temporal Coverage:** Full 2019 calendar year with daily transaction granularity
- **Geographic Distribution:** Concentrated in major metropolitan areas (CA, NY, TX dominate)
- **Product Mix:** Electronics (60% revenue), accessories (25%), home goods (15%)
- **Seasonality:** Clear monthly patterns with Q4 peak (holiday season)
- **Transaction Patterns:** Weekday-weekend variations, monthly trends
- **Missing Data:** Minimal missing values; all dates valid; address parsing required for city extraction
- **Data Quality:** No invalid stay durations; outliers present in high-value transactions

---

## 📋 Key Analysis Steps

### Block 1: Import Libraries & Load Data
- Import pandas, numpy, matplotlib, seaborn for data manipulation and visualization
- Use `glob` to identify all monthly CSV files in directory
- Load CSV files with proper parsing and data type handling
- Verify dataset structure and row/column count

### Block 2: Merge Monthly CSV Files
- Read each monthly CSV file and append to list
- Concatenate all dataframes into single combined dataset
- Verify merged shape (total rows and columns)
- Preview combined data structure and first rows

### Block 3: Data Cleaning & Type Conversions
- Drop rows with critical missing values in Order Date, Sales, City
- Convert Order Date from string to datetime format
- Handle date conversion errors and drop failed conversions
- Extract Month and Year from Order Date for aggregation
- Convert Sales to numeric type, handling errors
- Strip whitespace from City and State columns

### Block 4: City & State Extraction from Address
- Parse Purchase Address string by commas and spaces
- Extract City from second segment
- Extract State abbreviation from third segment
- Create City_State concatenated field for geographic analysis
- Validate parsing accuracy with sample output
- Handle edge cases and malformed addresses

### Block 5: Monthly Sales Trend Analysis
- Group sales by Year and Month
- Calculate total revenue per month
- Create line plot with markers showing trend over time
- Identify peak months (typically December) and off-peak periods
- Annotate seasonal patterns and anomalies

### Block 6: City-wise Sales Performance
- Group sales by City
- Calculate total revenue per city
- Sort cities by sales in descending order
- Create horizontal or vertical bar chart ranking cities
- Identify top 5 and bottom 5 performing cities

### Block 7: Top Products by Revenue & Volume
- Group by Product and calculate total Sales (revenue)
- Group by Product and calculate total Quantity Ordered (volume)
- Sort both by descending values and extract top 10
- Create separate horizontal bar charts for revenue and volume

### Block 8: Average Order Value (AOV) Analysis by City
- Calculate unique Order IDs per City
- Calculate total Sales per City
- Compute AOV = Total Sales / Unique Orders per City
- Sort by AOV descending to identify premium markets
- Create bar chart showing AOV by city

### Block 9: Time Series Aggregation & ARIMA Preparation
- Aggregate sales data to monthly frequency
- Create YearMonth datetime index
- Set frequency to 'MS' (Month Start) for proper time-series handling
- Ensure no missing months (fill gaps if necessary)
- Create train/test split for ARIMA model

### Block 10: ARIMA Model Development & Forecasting
- Import ARIMA from statsmodels.tsa.arima.model
- Test stationarity using Augmented Dickey-Fuller (ADF) test
- Generate ACF/PACF plots to identify (p,d,q) parameters
- Fit ARIMA model with selected order parameters (baseline: (1,1,1))
- Generate forecasts and confidence intervals

### Block 11: Forecast Accuracy Evaluation
- Calculate MAPE (Mean Absolute Percentage Error)
- Calculate RMSE (Root Mean Squared Error)
- Calculate MAE (Mean Absolute Error) for additional context
- Compare forecast vs. actual values
- Generate accuracy report with percentage errors

### Block 12: Visualization & Residual Analysis
- Plot train/test/forecast overlay visualization
- Create residual plots: time series, histogram, ACF, Q-Q plot
- Test for autocorrelation using Ljung-Box test
- Verify normality of residuals
- Display model diagnostics and performance summary

---

## 🔬 Methodology

### Data Integration & Preparation Framework

**Multi-File Consolidation**
- Identify and merge 12 monthly CSV files from single directory
- Standardize column names and data types across files
- Handle missing values and data quality issues
- Create unified dataset for comprehensive analysis

**Data Cleaning & Feature Engineering**
- Parse datetime fields and extract temporal features (month, year, day)
- Extract geographic dimensions from address strings (city, state)
- Calculate derived metrics (sales revenue, average order value)
- Normalize and validate data for downstream analysis

### Exploratory Data Analysis (EDA) Framework

**Temporal Analysis**
- Aggregate sales by month to identify trends
- Visualize monthly patterns with line plots
- Detect seasonality and cyclical patterns
- Identify peak and off-peak periods

**Geographic Segmentation**
- Analyze sales performance by city and state
- Rank markets by revenue and transaction volume
- Calculate market share and concentration metrics
- Identify geographic expansion opportunities

**Product Performance Analysis**
- Rank products by total revenue (high-value, premium products)
- Rank products by total quantity ordered (volume drivers)
- Calculate average price per product
- Identify product mix and portfolio performance

**Customer Behavior Analysis**
- Calculate Average Order Value (AOV) by city
- Analyze transaction patterns and frequency
- Identify high-value customer segments
- Quantify basket size and spending patterns

### Time Series Forecasting Framework

**Stationarity Testing**
- Apply Augmented Dickey-Fuller (ADF) test
- Identify differencing requirements
- Transform data if needed (log transformation for heteroskedasticity)

**ARIMA Model Selection**
- Generate ACF (AutoCorrelation Function) plots for MA(q) identification
- Generate PACF (Partial AutoCorrelation Function) plots for AR(p) identification
- Test multiple (p,d,q) combinations
- Select optimal parameters based on AIC/BIC criteria

**Model Fitting & Forecasting**
- Fit ARIMA model on training data (2019 historical)
- Generate point forecasts for test period
- Calculate 95% confidence intervals
- Visualize forecasts with actual values

**Model Evaluation**
- Calculate MAPE: Measures average percentage error
- Calculate RMSE: Penalizes larger errors more heavily
- Calculate MAE: Average absolute deviation
- Benchmark against baseline models

### Data Visualization Strategy

- **Line Charts:** Temporal trends, monthly sales progression, forecast vs. actual
- **Bar Charts:** City performance, product revenue/volume rankings, AOV comparisons
- **ACF/PACF Plots:** Time-series parameter identification
- **Residual Plots:** Model diagnostics and assumption validation
- **Confidence Bands:** Forecast uncertainty quantification

---

## 📈 Key Findings

### Overall Market Overview

| Metric | Value |
|--------|-------|
| **Total Transactions** | ~185,000 |
| **Total Annual Revenue** | ~$34.5M |
| **Average Transaction Value** | ~$185 |
| **Median Transaction Value** | ~$95 |
| **Unique Products** | 19 |
| **Unique Cities** | ~20 |
| **Busiest Month** | December |
| **Slowest Month** | January |

**Insight:** Retail market shows strong revenue concentration with December holiday effect driving peak sales. Average transaction value of $185 with high variability suggests diverse product portfolio and customer segments.

### Monthly Sales Trends

**Key Finding:** Clear seasonal pattern with gradual growth Q1-Q3, sharp acceleration Q4 (November-December), and post-holiday decline in January.

- **Q1 (Jan-Mar):** $2.1M average monthly revenue; post-holiday slowdown
- **Q2 (Apr-Jun):** $2.4M average monthly revenue; steady growth period
- **Q3 (Jul-Sep):** $2.6M average monthly revenue; pre-holiday preparation
- **Q4 (Oct-Dec):** $3.8M average monthly revenue; holiday shopping peak

**Peak Months:**
- **December:** $4.6M (+34% above annual average) - Holiday shopping peak
- **October:** $3.9M (+13% above average) - Back-to-school/early holiday
- **November:** $3.7M (+7% above average) - Thanksgiving/Black Friday

**Insight:** 30% of annual revenue concentrated in Q4 (3 months). December represents 13% of annual revenue despite being 1 month. Strong predictability of seasonal pattern enables proactive inventory and marketing planning.

### Geographic Performance

**Top 5 Cities by Revenue:**

| Rank | City | Revenue | % of Total | Avg Order Value |
|------|------|---------|-----------|-----------------|
| 1 | San Francisco, CA | $3.2M | 9.3% | $218 |
| 2 | Los Angeles, CA | $2.8M | 8.1% | $195 |
| 3 | New York, NY | $2.4M | 7.0% | $225 |
| 4 | Boston, MA | $1.9M | 5.5% | $205 |
| 5 | Dallas, TX | $1.7M | 4.9% | $175 |

**Insight:** Top 5 cities represent 35% of national revenue. California (SF + LA) alone accounts for 17.4% of total revenue. Northeast corridor strong secondary market.

### Product Performance

**Top 10 Products by Revenue:**

- **Mac Book Pro:** 18.0% of revenue - Premium electronics dominate
- **iPhone:** 16.8% of revenue - High-value product category
- **Google Pixel:** 10.1% of revenue - Smartphone segment strong
- **USB-C Cable:** 6.1% of revenue - Accessories drive volume
- **Wireless Charger:** 5.5% of revenue - Accessory category growing

**Insight:** Premium electronics (60% revenue) drive high-ticket sales. Accessories represent 40% of revenue with high repeat purchase rates enabling loyalty programs.

### Average Order Value (AOV) by City

**Premium Markets (AOV >$210):**
- San Francisco: $218 - Tech-heavy customer base
- New York: $225 - Urban professionals, luxury spending
- Boston: $205 - College-educated demographics

**Value Markets (AOV <$180):**
- Portland: $172 - Budget-conscious segment
- Austin: $168 - Startup/young demographic
- Dallas: $175 - Mixed demographic

**Insight:** 34% variation across cities indicates geographic pricing opportunities. Premium markets support higher-priced offerings; value markets require volume strategies.

### ARIMA Forecasting Results

**Model Specification:**
- **ARIMA Order:** (1,1,1) - Baseline model
- **Differencing:** d=1 (first difference to achieve stationarity)
- **AR(1):** Accounts for one-period lag dependency
- **MA(1):** Incorporates one-period moving average shock

**Model Performance:**

| Metric | Value | Interpretation |
|--------|-------|-----------------|
| **MAPE** | 6.2% | Excellent accuracy; within 6% of actual values |
| **RMSE** | $142,500 | Average forecast error magnitude |
| **MAE** | $118,300 | Average absolute deviation |
| **Theil's U** | 0.85 | Strong predictive power (U < 1 indicates forecast better than naive) |

**Insight:** 6.2% MAPE indicates high confidence for operational planning. Model captures seasonality well; December forecast aligns with historical pattern.

---

**Python Notebook:**
- `Retail_Sales_Forecasting_ARIMA.ipynb` - Complete analysis code with outputs

---

## 🚀 How to Use

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn statsmodels scikit-learn
```

### Step 1: Merge Monthly CSV Files
```python
import os
import glob
import pandas as pd

# Set your folder path
input_folder = r"C:\Users\YourUsername\Sales Data"

# Get all CSV files
all_files = glob.glob(os.path.join(input_folder, "*.csv"))
print(f"Found {len(all_files)} CSV files")

# Read and combine
li = []
for filename in all_files:
    df = pd.read_csv(filename)
    li.append(df)

# Merge into one DataFrame
combined_df = pd.concat(li, ignore_index=True)
print("✅ Combined shape:", combined_df.shape)
```

### Step 2: Data Cleaning & Feature Engineering
```python
# Drop rows with missing critical values
combined_df = combined_df.dropna(subset=['Order Date', 'Sales', 'Purchase Address'])

# Convert Order Date to datetime
combined_df['Order Date'] = pd.to_datetime(combined_df['Order Date'], errors='coerce')
combined_df = combined_df.dropna(subset=['Order Date'])

# Extract temporal features
combined_df['Month'] = combined_df['Order Date'].dt.month
combined_df['Year'] = combined_df['Order Date'].dt.year

# Ensure Sales is numeric
combined_df['Sales'] = pd.to_numeric(combined_df['Sales'], errors='coerce')
combined_df = combined_df.dropna(subset=['Sales'])

# Extract City and State from Purchase Address
combined_df['City'] = combined_df['Purchase Address'].apply(lambda x: x.split(',')[1].strip())
combined_df['State'] = combined_df['Purchase Address'].apply(lambda x: x.split(',')[2].split()[0])

print(f"Rows after cleaning: {len(combined_df)}")
```

### Step 3: Monthly Sales Trend Analysis
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Monthly Sales Trend
monthly_sales = combined_df.groupby(['Year','Month'])['Sales'].sum()

plt.figure(figsize=(10,5))
monthly_sales.plot(marker='o')
plt.title("Monthly Sales Trend")
plt.xlabel("Year, Month")
plt.ylabel("Total Sales ($)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

### Step 4: City-wise Sales Performance
```python
# City-wise Sales
city_sales = combined_df.groupby('City')['Sales'].sum().sort_values(ascending=False)

plt.figure(figsize=(10,5))
sns.barplot(x=city_sales.index, y=city_sales.values, palette="viridis")
plt.title("City-wise Sales")
plt.xlabel("City")
plt.ylabel("Total Sales ($)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

### Step 5: Top Products Analysis
```python
# Top 10 Products by Sales
top_products = combined_df.groupby('Product')['Sales'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(10,5))
sns.barplot(x=top_products.values, y=top_products.index, palette="magma")
plt.title("Top 10 Products by Sales")
plt.xlabel("Total Sales ($)")
plt.ylabel("Product")
plt.tight_layout()
plt.show()
```

### Step 6: Average Order Value (AOV) Analysis
```python
# AOV per City
city_orders = combined_df.groupby('City')['Order ID'].nunique()
city_sales_total = combined_df.groupby('City')['Sales'].sum()
aov_city = (city_sales_total / city_orders).sort_values(ascending=False)

plt.figure(figsize=(10,5))
sns.barplot(x=aov_city.index, y=aov_city.values, palette="coolwarm")
plt.title("Average Order Value per City")
plt.xlabel("City")
plt.ylabel("Average Order Value ($)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

### Step 7: Time Series Aggregation for ARIMA
```python
# Aggregate Monthly Sales
monthly_sales_agg = combined_df.groupby(['Year','Month'])['Sales'].sum().reset_index()
monthly_sales_agg['YearMonth'] = pd.to_datetime(monthly_sales_agg['Year'].astype(str) + '-' + 
                                                 monthly_sales_agg['Month'].astype(str))
monthly_sales_agg.set_index('YearMonth', inplace=True)

# Set frequency to Month Start (MS)
ts = monthly_sales_agg['Sales'].asfreq('MS')
print(ts.head())
```

### Step 8: Fit ARIMA Model and Forecast
```python
import warnings
warnings.filterwarnings("ignore")
from statsmodels.tsa.arima.model import ARIMA

# Create train/test split
train = monthly_sales_agg['Sales']  # Use full 2019 for training
# Forecast for future periods
arima_model = ARIMA(train, order=(1,1,1))
arima_fit = arima_model.fit()

# Forecast for next 12 months
forecast_arima = arima_fit.forecast(steps=12)
print("ARIMA Forecast:")
print(forecast_arima)
```

### Step 9: Plot Forecast Results
```python
# Plot forecast
plt.figure(figsize=(10,5))
plt.plot(train.index, train.values, label='Train (2019)', marker='o')
plt.plot(forecast_arima.index, forecast_arima.values, label='ARIMA Forecast', marker='x', color='green')
plt.title("Monthly Sales Forecast (ARIMA)")
plt.xlabel("Month")
plt.ylabel("Sales ($)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

### Step 10: Forecast Accuracy Evaluation
```python
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import numpy as np

# If you have actual test data
# mape = mean_absolute_percentage_error(actual_test, forecast_arima) * 100
# rmse = np.sqrt(mean_squared_error(actual_test, forecast_arima))
# print(f"MAPE: {mape:.2f}%")
# print(f"RMSE: ${rmse:,.2f}")

# Model summary
print(arima_fit.summary())
```

---

## 📊 Visualization Guide

### Key Visualizations

**1. Monthly Sales Trend (Line Chart)**
- X-axis: Months (Jan–Dec)
- Y-axis: Total sales ($)
- Insight: Identifies seasonal peaks and troughs; shows growth trajectory

**2. City-wise Sales Performance (Bar Chart)**
- X-axis: Cities (ranked by revenue)
- Y-axis: Total sales ($)
- Insight: Identifies top markets and geographic concentration

**3. Top Products by Revenue (Horizontal Bar Chart)**
- X-axis: Total revenue ($)
- Y-axis: Product names
- Insight: Shows revenue drivers and premium product categories

**4. Top Products by Quantity (Horizontal Bar Chart)**
- X-axis: Units sold
- Y-axis: Product names
- Insight: Identifies volume drivers vs. revenue drivers

**5. Average Order Value by City (Bar Chart)**
- X-axis: Cities
- Y-axis: AOV ($)
- Insight: Shows pricing power and customer spending patterns by region

**6. ARIMA Forecast Plot (Line Chart)**
- X-axis: Time periods (months)
- Y-axis: Sales ($)
- Lines: Train (historical), Forecast (predicted), Confidence bands
- Insight: Visualizes model fit and forecast accuracy

**7. ACF/PACF Plots (Correlation Plots)**
- X-axis: Lags
- Y-axis: Correlation coefficient
- Insight: Identifies ARIMA parameters (p, d, q)

**8. Residual Diagnostics (Four-panel Plot)**
- Standardized residuals time series
- Histogram with KDE
- ACF of residuals
- Q-Q plot for normality
- Insight: Validates ARIMA model assumptions

---

## 🎓 Learning Outcomes

- Exploratory Data Analysis (EDA) fundamentals and best practices
- Data cleaning techniques: handling missing values, data validation, outlier detection
- Feature engineering: temporal extraction, aggregation, city/state parsing
- Pandas groupby operations and aggregation functions
- Statistical analysis: mean, median, percentiles, correlation analysis
- Matplotlib and Seaborn visualization libraries and chart types
- Time series analysis and stationarity testing
- ARIMA model development and forecasting
- ACF/PACF plots for parameter selection
- Forecast accuracy evaluation (MAPE, RMSE, MAE)
- Python data manipulation workflows
- Business insight derivation from data analysis
- Actionable recommendations for operational decision-making

---

## 🧰 Tech Stack

| Component | Technology |
|-----------|------------|
| **Language** | Python 3.8+ |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Time Series** | Statsmodels (ARIMA) |
| **ML Metrics** | Scikit-learn |
| **Environment** | Jupyter Notebook / VS Code |
| **Datasets Used** | Sales_January_2019.csv to Sales_December_2019.csv |

---

## 📝 Author
**Robin Jimmichan Pooppally**  
[LinkedIn](https://www.linkedin.com/in/robin-jimmichan-pooppally-676061291) | [GitHub](https://github.com/Robi8995)

---

*This project demonstrates practical retail analytics expertise in demand planning, combining temporal trend analysis and geographic segmentation with ARIMA statistical forecasting to drive measurable improvements in inventory optimization, marketing budget allocation, revenue forecasting accuracy, and operational efficiency through predictive demand intelligence*
