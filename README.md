# Walmart-Sales-Forecast-Analytical-Dashboard
An end-to-end ML solution predicting weekly retail sales across 45 stores. Developed a robust pipeline using MySQL and Python to process economic indicators and seasonal trends. Evaluated ARIMA, LSTM, and XGBoost, with LightGBM achieving a 4.11% MAPE. Features a Streamlit dashboard for real-time forecasting and automated data-driven decision-making



```markdown
# 🛒 Walmart Sales Forecasting (End-to-End ML Project)

An end-to-end Time Series Forecasting project that predicts Walmart weekly sales using Machine Learning and Deep Learning models. The project includes data analysis, feature engineering, model building, and deployment using Streamlit.

---

## 🚀 Project Overview

Retail sales forecasting is critical for inventory management, staffing, and business strategy. This project uses historical Walmart sales data along with external factors to predict future weekly sales.

The solution compares multiple models and deploys the best-performing model as an interactive web application.

---

## 📊 Dataset

- Source: Kaggle Walmart Sales Dataset
- Stores: 45
- Time Period: 2010 – 2012
- Total Records: ~6,400

### Features Used:
- Store information
- Holiday flag
- Temperature
- Fuel price
- CPI
- Unemployment
- Date-based features (Year, Month, Week)
- Lag features (previous sales)
- Rolling statistics

---

## 🧠 Models Implemented

| Model        | Type              | Notes |
|-------------|------------------|------|
| ARIMA       | Statistical      | Applied on single store |
| XGBoost     | ML (Boosting)    | Good performance |
| LightGBM ⭐ | ML (Boosting)    | Best overall model |
| LSTM        | Deep Learning    | Higher error, needs more tuning |

---

## 🏆 Best Model Performance (LightGBM)

- MAE: ~38,348  
- RMSE: ~60,019  
- MAPE: ~4.11%  

---

## ⚙️ Feature Engineering

Key features created:

- Lag Features: Sales_Lag_1, 2, 3, 52
- Rolling Features: 4-week mean & std
- Sales Velocity
- Fuel Price Change
- Pre-Holiday Indicator
- Payday Week Indicator
- Store Average Sales

---

## 📈 Exploratory Data Analysis

Performed:

- Sales distribution & outlier detection
- Store-wise sales comparison
- Holiday vs non-holiday analysis
- Seasonality and trend analysis
- Correlation heatmap
- Time series decomposition

---

## 🖥️ Application (Streamlit)

An interactive dashboard with:

### 🔮 Sales Prediction
- Input business parameters
- Predict weekly sales
- Compare with store average

### 📊 Store Analysis
- Sales trends per store
- Holiday impact visualization
- Statistical summary

### 📋 Model Comparison
- Performance comparison across models
- Feature importance overview

---

## 🛠️ Tech Stack

- **Language:** Python
- **Libraries:** pandas, numpy, sklearn, lightgbm, xgboost, tensorflow
- **Visualization:** matplotlib, seaborn
- **Database:** MySQL
- **Deployment:** Streamlit

---

## 📂 Project Structure

```

├── app_forecast.py              # Streamlit app
├── walmart_forecasting.py      # EDA + Model training
├── walmart_forecast_model.pkl  # Saved model
├── requirements.txt
├── plots_forecasting/
└── dataset/

````

---

## ▶️ How to Run the Project

### 1. Clone Repository
```bash
git clone <your-repo-link>
cd project-folder
````

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Setup MySQL Database

* Create database: `forecasting`
* Import dataset into table: `sales`

### 4. Run Application

```bash
streamlit run app_forecast.py
```

---

## ⚠️ Limitations

* LSTM model not fully optimized
* ARIMA applied only to single store
* Depends on MySQL (can be replaced with CSV for simplicity)

---


## 👨‍💻 Author

**Abenesh**
- GitHub: [@abenesh123](https://github.com/abenesh123)

---

