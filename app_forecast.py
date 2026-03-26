import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Walmart Sales Forecast",
    page_icon="🛒",
    layout="wide"
)

@st.cache_resource
def load_model():
    model = joblib.load("walmart_forecast_model.pkl")
    print(model.feature_name_)
    return model

@st.cache_data
def load_data():
    df = pd.read_csv("walmart-sales-dataset-of-45stores.csv")
    df["Date"]=pd.to_datetime(df["Date"],format="%d-%m-%Y")
    return df

model=load_model()

st.title("🛒 Walmart Weekly Sales Forecast")
st.markdown("Predict Future Weekly Sales Using Machine learning - LGBM,XGB,LSTM,ARIMA")
st.divider()

with st.sidebar:
    st.header("ℹ️ ABOUT")
    st.write("This APP Forecasts Walmart Weekly Sales Using Historical and External Factors")
    st.divider()
    st.header("📊 Model Info")
    st.write("**Best Model:** LightGBM")
    st.write("**MAE:**  ~38,348")
    st.write("**RMSE:** ~60,019")
    st.write("**MAPE:** ~4.11%")
    st.divider()
    st.header("📁 Dataset")
    st.write("**Source:** Kaggle Walmart Sales")
    st.write("**Stores:** 45")
    st.write("**Period:** 2010–2012")
    st.write("**Records:** 6,435")


tab1,tab2,tab3=st.tabs(["🔮 Predict Sales", "📈 Store Analysis", "📋 Model Comparison"])

with tab1:
    st.subheader("Enter Transaction Details")

    col1,col2,col3=st.columns(3)

    with col1:
        store = st.selectbox("Store Number",options=list(range(1,46)))
        holiday_flag=st.selectbox("Holiday Week?",options=[0,1],format_func=(lambda x:"YES" if x==1 else "NO"))
        temperature=st.number_input("Temperature (^F)",value=60.0)
        fuel_price=st.number_input("Fuel Price ($)",value=3.5)

    with col2:
        cpi=st.number_input("CPI",value=210.0) 
        unemployment=st.number_input("Unemployment Rate (%)",value=8.0)
        month=st.selectbox("Month",options=list(range(1,13)),format_func=lambda x:["Jan","Feb","Mar","Apr","May","Jun",
                                                                                   "Jul","Aug","Sep","Oct","Nov","Dec"][x-1])
        year=st.selectbox("Year",options=[2010,2011,2012,2013])

    with col3:
        week_of_year=st.number_input("Week Of Year",min_value=1,max_value=52,value=20)
        day_of_month=st.number_input("Day Of Month",min_value=1,max_value=31,value=5)
        is_payday=st.selectbox("Is Payday?",options=[0,1],format_func=lambda x:"YES" if x==1 else "NO")  
        pre_holiday_week=st.selectbox("Is Pre-Holiday Week",options=[0,1],format_func=lambda x:"YES" if x==1 else "NO")

    st.divider()   
    st.subheader("Lag Features (previous sales)")

    col4,col5=st.columns(2)


    with col4:
        sales_lag_1=st.number_input("Sales 1 Week Ago",value=1000000.0)
        sales_lag_2=st.number_input("Sales 2 Week Ago",value=980000.0)
        sales_lag_3=st.number_input("Sales 3 Week Ago",value=960000.0)
        sales_lag_52=st.number_input("Sales 52 Week Ago (same week last year)",value=1050000)


    with col5:
        rolling_mean_4=st.number_input("4-week Rolling Mean",value=990000.0) 
        rolling_std_4=st.number_input("4-Week Rolling Std",value=50000.0) 
        fuel_price_change= st.number_input("Fuel Price Change",      value=0.05)
        store_avg_sales  = st.number_input("Store Average Sales",    value=1046964.0)


    if st.button("🔮 Predict Weekly Sales", use_container_width=True):
    
        sales_velocity = (sales_lag_1 - sales_lag_2) / (sales_lag_2 + 1e-5)
    
        input_data = pd.DataFrame([[
        store, holiday_flag, temperature, fuel_price, cpi, unemployment,
        year, month, week_of_year, day_of_month,
        is_payday, sales_lag_1, sales_lag_2, sales_lag_3, sales_lag_52,
        rolling_mean_4, rolling_std_4, sales_velocity,
        fuel_price_change, pre_holiday_week, store_avg_sales
        ]], columns=[
        'Store', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment',
        'Year', 'Month', 'Week_of_Year', 'Day_of_Month',
        'Is_Payday_Week', 'Sales_Lag_1', 'Sales_Lag_2', 'Sales_Lag_3', 'Sales_Lag_52',
        'Rolling_Mean_4', 'Rolling_Std_4', 'Sales_Velocity',
        'Fuel_Price_Change', 'Pre_Holiday_Week', 'Store_Avg_Sales'
        ])
    
        prediction = model.predict(input_data)[0]    

        st.divider()

        col_pred,col_info = st.columns(2)

        with col_pred:
           

           st.metric(
            label="Prediction Weekly Sales",
            value=(f"${prediction:.2f}"),
            delta=(f"{((prediction-store_avg_sales)/store_avg_sales*100):.1f} vs store avg")
        )


        with col_info:
           if prediction>store_avg_sales*1.1:
              
              st.success("🔼 HIGH SALES WEEK — Above average by 10%+")  
           elif prediction<store_avg_sales*0.9:
              st.warning("🔽 LOW SALES WEEK — Below average by 10%+")
           else:
              st.info("➡️ NORMAL SALES WEEK — Within average range")   


        st.divider()
        st.subheader("Input Summary")
        summary = pd.DataFrame({
            "Feature": ["Store", "Holiday Week", "Temperature", "Fuel Price",
                        "CPI", "Unemployment", "Month", "Is Payday Week",
                        "Sales Last Week", "Store Avg Sales"],
            "Value": [store, "Yes" if holiday_flag else "No", temperature,
                      fuel_price, cpi, unemployment, month,
                      "Yes" if is_payday else "No",
                      f"{sales_lag_1:,.0f}", f"{store_avg_sales:,.0f}"]
        })
        st.dataframe(summary, use_container_width=True)
      

with tab2:
    st.subheader("Store Sales Analysis")
 
    try:
        df = load_data()
 
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Records",    f"{len(df):,}")
        col2.metric("Total Stores",     "45")
        col3.metric("Avg Weekly Sales", f"{df['Weekly_Sales'].mean():,.0f}")
        col4.metric("Max Weekly Sales", f"{df['Weekly_Sales'].max():,.0f}")
 
        st.divider()
 
        # Store selector
        selected_store = st.selectbox("Select Store to Analyze", options=list(range(1, 46)))
 
        store_df = df[df["Store"] == selected_store].sort_values("Date")
 
        col_a, col_b = st.columns(2)
 
        with col_a:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(store_df["Date"], store_df["Weekly_Sales"], color="steelblue")
            ax.set_title(f"Store {selected_store} — Weekly Sales Over Time")
            ax.set_xlabel("Date")
            ax.set_ylabel("Weekly Sales")
            plt.tight_layout()
            st.pyplot(fig)
 
        with col_b:
            holiday_avg = store_df.groupby("Holiday_Flag")["Weekly_Sales"].mean()
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            bars = ax2.bar(["Non-Holiday", "Holiday"],
                           [holiday_avg.get(0, 0), holiday_avg.get(1, 0)],
                           color=["steelblue", "coral"], width=0.4)
            ax2.set_title(f"Store {selected_store} — Holiday vs Non-Holiday Sales")
            ax2.set_ylabel("Avg Weekly Sales")
            for bar in bars:
                ax2.text(bar.get_x() + bar.get_width()/2,
                         bar.get_height() + 5000,
                         f'{bar.get_height():,.0f}',
                         ha='center', fontsize=9)
            plt.tight_layout()
            st.pyplot(fig2)
 
        # Store stats
        st.divider()
        st.subheader(f"Store {selected_store} Statistics")
        stats = store_df["Weekly_Sales"].describe()
        stats_df = pd.DataFrame({
            "Metric": ["Count", "Mean", "Std Dev", "Min", "25%", "Median", "75%", "Max"],
            "Value":  [f"{stats['count']:.0f}",
                       f"{stats['mean']:,.2f}",
                       f"{stats['std']:,.2f}",
                       f"{stats['min']:,.2f}",
                       f"{stats['25%']:,.2f}",
                       f"{stats['50%']:,.2f}",
                       f"{stats['75%']:,.2f}",
                       f"{stats['max']:,.2f}"]
        })
        st.dataframe(stats_df, use_container_width=True)
 
    except Exception as e:
        st.error(f"Could not load data from MySQL: {e}")
        st.info("Make sure MySQL is running and the forecasting database exists.")


with tab3:
    st.subheader("Model Performance Comparision ")

    comparision_table=pd.DataFrame({
        "Model":  ["ARIMA (Store 1)", "XGBoost", "LightGBM ⭐", "LSTM"],
        "MAE":    ["—",    "38,582",  "38,348",  "60,121"],
        "RMSE":   ["—",    "59,455",  "60,019",  "78,004"],
        "MAPE":   ["—",    "4.24%",   "4.11%",   "7.14%"],
        "Type":   ["Statistical", "Tree-based", "Tree-based", "Deep Learning"]
    ,

    "Notes":["Store 1 only — univariate",
            "Good balance of speed and accuracy",
            "Best overall — lowest MAPE",
            "Higher error — needs more data"]
    })

    st.dataframe(comparision_table,use_container_width=True)

    st.divider()

    st.subheader("Why LightGBM Wins")
    col1,col2,col3=st.columns(3)

    col1.metric("MAPE",  "4.11%",  "Best")
    col2.metric("MAE",  "38,348",  "Lower")
    col3.metric("Speed",  "Fast",  "vs LSTM")


    st.divider()

    st.subheader("Key Features Used")

    features_info=pd.DataFrame({
        "Feature Category":["Store Info", "Time Features", "Holiday Features",
            "Lag Features", "Rolling Features", "External Factors"],

        "Features":["Store number, Store avg sales",
            "Year, Month, Week of Year, Day of Month",
            "Holiday Flag, Pre-Holiday Week",
            "Sales Lag 1, 2, 3, 52 weeks",
            "Rolling Mean 4 weeks, Rolling Std 4 weeks",
            "Temperature, Fuel Price, CPI, Unemployment"]    
    })

    st.dataframe(features_info,use_container_width=True)

    st.divider()


    st.subheader("Tech Stack")

    col1,col2=st.columns(2)

    with col1:
        st.write("**ML Models:** LightGBM, XGBoost, LSTM, ARIMA")
        st.write("**Database:** MySQL")
        st.write("**Language:** Python 3.x")

    with col2:
        st.write("**Libraries:** pandas, numpy, sklearn, tensorflow")
        st.write("**Deployment:** Streamlit Cloud")
        st.write("**Dataset:** Kaggle Walmart Sales (45 stores)")

