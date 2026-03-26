import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings("ignore")

engine=create_engine("mysql+mysqlconnector://root:abineshmysql%40123@localhost/forecasting")

df=pd.read_sql("SELECT * FROM sales",con=engine)
print("Data loaded successfully")

print(df)

print(df.head())
print(df.columns)
print(df.shape)
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df.duplicated().sum())
print(df["Store"].nunique())
print(f"start:{df["Date"].min()} | End:{df['Date'].max()}")

print("Basic check completed")


plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


plt.figure(figsize=(15,5))
sns.histplot(df["Weekly_Sales"],bins=50,kde=True)
plt.xlabel("Sales")
plt.ylabel("Frequency")
plt.title("Weekly Sales distribution")
plt.show()

plt.figure(figsize=(15,5))
sns.boxplot(x=df["Weekly_Sales"])
plt.title("Weekly Sales outlier detection")
plt.show()

Q1=df["Weekly_Sales"].quantile(0.25)
Q3=df["Weekly_Sales"].quantile(0.75)

IQR=Q3-Q1

lower=Q1-1.5*IQR
higher=Q3+1.5*IQR

outliers=df[(df["Weekly_Sales"]<lower) | (df["Weekly_Sales"]>higher)]

print(len(outliers))
print(f"lower:{lower} | higher:{higher}")

print("Sales Distribution Completed")

Weekly_total=df.groupby("Date")["Weekly_Sales"].sum().reset_index()
plt.figure(figsize=(15,5))
plt.plot(Weekly_total["Date"],Weekly_total["Weekly_Sales"],color="steelblue")
plt.xlabel("Date")
plt.ylabel("total weekly Sales")
plt.title("total weekly sales over time")
plt.show()

store1=df[df["Store"]==1].copy()
plt.figure(figsize=(15,5))
plt.plot(store1["Date"],store1["Weekly_Sales"],color="darkorange")
plt.xlabel("Date")
plt.ylabel("weekly sales")
plt.title("weekly sales over time for store 1")
plt.show()

store1=store1.set_index("Date").sort_index()
store1["MA_4"]=store1["Weekly_Sales"].rolling(window=4).mean()
store1["MA_12"]=store1["Weekly_Sales"].rolling(window=12).mean()

plt.figure(figsize=(15,5))
plt.plot(store1.index,store1["Weekly_Sales"],label="weekly_sales",color="yellow")
plt.plot(store1.index,store1["MA_4"],label="4-week-MA",color="red")
plt.plot(store1.index,store1["MA_12"],label="12-week-MA",color="blue")
plt.xlabel("Date")
plt.ylabel("weekly sales")
plt.title("store 1- sales with moving average")
plt.legend()
plt.show()

print("Time series visualization completed")

store_sales=df.groupby("Store")["Weekly_Sales"].sum().reset_index()
store_sales=store_sales.sort_values("Weekly_Sales",ascending=False)

plt.figure(figsize=(20,6))
sns.barplot(data=store_sales,x="Store",y="Weekly_Sales")
plt.xlabel("Stores")
plt.ylabel("Weekly sales")
plt.title("Weekly Sales For Each Stores")
plt.show()

top_5=store_sales.head(5)
bottom_5=store_sales.tail(5)

fig,axes=plt.subplots(1,2,figsize=(15,5))
sns.barplot(data=top_5,x="Store",y="Weekly_Sales",ax=axes[0],palette="Greens_r")
axes[0].set_title("top 5 stores sales")
sns.barplot(data=bottom_5,x="Store",y="Weekly_Sales",ax=axes[1],palette="Reds_r")
axes[1].set_title("bottom 5 stores sales")

plt.tight_layout()
plt.show()

plt.figure(figsize=(25,6))
sns.boxplot(data=df,x="Store",y="Weekly_Sales")
plt.xlabel("Store")
plt.ylabel("Weekly Sales")
plt.title("Sales Distribution Across All Stores")
plt.show()

print("Store Analysis Completed")

holiday_avg=df.groupby("Holiday_Flag")["Weekly_Sales"].mean().reset_index()
holiday_avg["Holiday_Flag"]=holiday_avg["Holiday_Flag"].map({0:"non-holiday",1:"holiday"})

plt.figure(figsize=(15,6))
sns.barplot(data=holiday_avg,x="Holiday_Flag",y="Weekly_Sales",palette=["steelblue","coral"],width=0.3)
plt.xlabel("Holiday_Flag")
plt.ylabel("Weekly Sales")
plt.title("Average Weekly Sales According to Holiday Flag")
plt.show()

print(df.groupby("Holiday_Flag")["Weekly_Sales"].agg(["mean","sum","count"]))

print("Holiday impact visualization Completed")

df["month"]=df["Date"].dt.month
df["year"]=df["Date"].dt.year

monthly_avg=df.groupby("month")["Weekly_Sales"].mean().reset_index()
month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

monthly_avg["month_names"]=monthly_avg["month"].apply(lambda x: month_names [x-1])

plt.figure(figsize=(15,5))
sns.barplot(data=monthly_avg,x="month_names",y="Weekly_Sales",palette="Blues_r")
plt.xlabel("months")
plt.ylabel("Average Weekly Sales")
plt.title("Monthly Average Of Weekly Sales")
plt.show()

yearly_sales=df.groupby("year")["Weekly_Sales"].sum().reset_index()

plt.figure(figsize=(15,5))
sns.barplot(data=yearly_sales,x="year",y="Weekly_Sales",palette="Reds_r",width=0.3)
plt.xlabel("Year")
plt.ylabel("Weekly Sales")
plt.title("Yearly Sales")
plt.show()

pivot=df.groupby([df["Date"].dt.year,df["Date"].dt.month])["Weekly_Sales"].mean().unstack()
pivot.index.name="year"
pivot.columns=month_names
plt.figure(figsize=(15,5))
sns.heatmap(pivot,cmap="YlOrRd",annot=True,fmt=".0f")
plt.title("Average Weekly Sales Heatmap (year,month)")
plt.show()

print("Seasonality Visualization Completed")

features = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']

fig,axes=plt.subplots(2,2,figsize=(15,5))
axes=axes.flatten()

for i,feature in enumerate(features):
    axes[i].scatter(df[feature],df["Weekly_Sales"],alpha=0.3,color="steelblue",s=5)
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel("Weekly Sales")
    axes[i].set_title(f"{feature} vs Weekly Sales")

plt.suptitle("External Factors vs Weekly Sales",fontsize=14) 
plt.tight_layout()
plt.show()   

print("External Factor Visualization Completed")

numeric_cols = ['Weekly_Sales', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
corr = df[numeric_cols].corr()

plt.figure(figsize=(15,5))
sns.heatmap(corr,annot=True,cmap="coolwarm",fmt=".2f",linewidths=0.5)
plt.title("Corelation Heatmap")
plt.show()

print(corr["Weekly_Sales"].sort_values(ascending=False))

print("Corelation Heatmap Completed")

store1_ts = df[df["Store"]==1].set_index("Date")["Weekly_Sales"].sort_index()

decompose=seasonal_decompose(store1_ts,model="additive",period=52)

fig,axes=plt.subplots(4,1,figsize=(15,5))
decompose.observed.plot(ax=axes[0],title="Observed")
decompose.trend.plot(ax=axes[1],title="Trend")
decompose.seasonal.plot(ax=axes[2],title="Seasonal")
decompose.resid.plot(ax=axes[3],title="Residual")
plt.suptitle("Store 1 - Time Series Decomposition",fontsize=14)
plt.tight_layout()
plt.show()

store1_df=df[df["Store"]==1]
sns.boxplot(data=store1_df,x="Holiday_Flag",y="Weekly_Sales",palette=["steelblue","red"])
plt.title("Holiday Flag sales outlier")
plt.xticks ([0,1],["Non-Holiday","Holiday"])
plt.show()

print("Store 1- Decomposition Completed")

print("EDA SUMMARY")
print(f"Total records: {len(df)}")
print(f"Date Range: Start:{df["Date"].min().date()} | End:{df["Date"].max().date()}")
print(f"Avg weekly sales:{df["Weekly_Sales"].mean()}")
print(f"Max weekly sales:{df["Weekly_Sales"].max()}")
print(f"Min weekly sales:{df["Weekly_Sales"].min()}")
print(f"Holiday Weeks:{df[df["Holiday_Flag"]==1]["Date"].nunique()}")
print(f"Non-Holiday Week:{df[df["Holiday_Flag"]==0]["Date"].nunique()}")
print(f"Outliers:{len(outliers)}")
print(f"Top_5_Stores:{top_5}")
print(f"Bottom_5_Stores:{bottom_5}")

df=df.drop(columns=["month","year"])

df["Year"]=df["Date"].dt.year
df["Month"]=df["Date"].dt.month
df["Week_of_Year"]=df["Date"].dt.isocalendar().week.astype(int)
df["Day_of_Month"]=df["Date"].dt.day

df["Is_Payday_Week"]=df["Day_of_Month"].apply(lambda x:1 if x<=7 or(x>=15 and x<=22) else 0)


for lag in [1,2,3,52]:
    df[f"Sales_Lag_{lag}"]=df.groupby("Store")["Weekly_Sales"].shift(lag)


df["Rolling_Mean_4"]=df.groupby("Store")["Weekly_Sales"].transform(lambda x:x.shift(1).rolling(window=4).mean())
df["Rolling_Std_4"]=df.groupby("Store")["Weekly_Sales"].transform(lambda x:x.shift(1).rolling(window=4).std())

df["Sales_Velocity"]=(df["Sales_Lag_1"]-df["Sales_Lag_2"])/(df["Sales_Lag_2"]+1e-5)

df["Fuel_Price_Change"]=df.groupby("Store")["Fuel_Price"].diff()

df["Pre_Holiday_Week"]=df.groupby("Store")["Holiday_Flag"].shift(-1).fillna(0)

store_avg_sales=df.groupby("Store")["Weekly_Sales"].mean().to_dict()
df["Store_Avg_Sales"]=df["Store"].map(store_avg_sales)

df_ML=df.dropna().reset_index(drop=True)
print(f"Feature Created New Shape {df_ML.shape}")
print(df_ML.columns)

df_ML=df_ML.sort_values("Date")

X=df_ML.drop(["Date","Weekly_Sales"],axis=1)
Y=df_ML["Weekly_Sales"]

split_index=int(len(df_ML)*0.8)

X_train=X.iloc[:split_index]
X_test=X.iloc[split_index:]
Y_train=Y.iloc[:split_index]
Y_test=Y.iloc[split_index:]

print(f"Training Data Range :{df_ML.iloc[0]["Date"]} to {df_ML.iloc[split_index-1]["Date"]}")
print(f"Testing Data Range :{df_ML.iloc[split_index]["Date"]} to {df_ML.iloc[-1]["Date"]}")


from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

store1_ts=df_ML[df_ML["Store"]==1].set_index("Date")["Weekly_Sales"].sort_index()
train_st1=store1_ts.iloc[:int(len(store1_ts)*0.8)]
test_st1=store1_ts.iloc[int(len(store1_ts)*0.8):]

adf_result=adfuller(train_st1)
print(f"ADF Statistic :{adf_result[0]:.4f}")
print(f"P-Value :{adf_result[1]:.4f}")

if adf_result[1] < 0.05:
    print("Series is Stationary")
else:
    print("Series is NOT Stationary — differencing needed")

ARIMA_Model=ARIMA(train_st1,order=(5,1,0))
ARIMA_Fit=ARIMA_Model.fit()
ARIMA_Pred=ARIMA_Fit.forecast(steps=len(test_st1))



# SARIMAX_Model=SARIMAX(train_st1,order=(1,1,1),seasonal_order=(1,1,1,52))
# SARIMAX_Fit=SARIMAX_Model.fit(disp=False)
# SARIMAX_Pred=SARIMAX_Fit.forecast(steps=len(test_st1))


import lightgbm as lgb 
from xgboost import XGBRegressor

XGB_Model=XGBRegressor(n_estimators=1000,learning_rate=0.05,max_depth=6,early_stopping_rounds=50,verbose=False)
XGB_Model.fit(X_train,Y_train,eval_set=[(X_test,Y_test)])
XGB_Pred=XGB_Model.predict(X_test)

LGBM_Model=lgb.LGBMRegressor(n_estimators=1000,learning_rate=0.05,verbose=-1)
LGBM_Model.fit(X_train,Y_train,eval_set=[(X_test,Y_test)],eval_metric="rmse",callbacks=[lgb.early_stopping(50)])
LGBM_Pred=LGBM_Model.predict(X_test)

from sklearn.preprocessing import MinMaxScaler
scaler_X=MinMaxScaler()
scaler_Y=MinMaxScaler()

X_train_scale=scaler_X.fit_transform(X_train)
X_test_scale=scaler_X.transform(X_test)

Y_train_scale=scaler_Y.fit_transform(Y_train.values.reshape(-1,1))
Y_test_scale=scaler_Y.transform(Y_test.values.reshape(-1,1))

X_train_3D=X_train_scale.reshape(X_train_scale.shape[0],1,X_train_scale.shape[1])
X_test_3D=X_test_scale.reshape(X_test_scale.shape[0],1,X_test_scale.shape[1])

print(f"LSTM Input Shape :{X_train_3D.shape}")

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dropout,Dense

model=Sequential([
    LSTM(64,activation="relu",input_shape=(X_train_3D.shape[1],X_train_3D.shape[2]),return_sequences=True),
    Dropout(0.2),
    LSTM(32,activation="relu"),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer="adam",loss="mse")

history=model.fit(
    X_train_3D,Y_train_scale,
    epochs=30,
    batch_size=32,
    validation_data=(X_test_3D,Y_test_scale),
    verbose=1
)

LSTM_Pred_scale=model.predict(X_test_3D)
LSTM_Pred=scaler_Y.inverse_transform(LSTM_Pred_scale)
Y_test_actual=scaler_Y.inverse_transform(Y_test_scale)

from sklearn.metrics import mean_absolute_error,root_mean_squared_error,mean_absolute_percentage_error

ARIMA_MAE=mean_absolute_error(test_st1,ARIMA_Pred)
ARIMA_RMSE=root_mean_squared_error(test_st1,ARIMA_Pred)
ARIMA_MAPE=mean_absolute_percentage_error(test_st1,ARIMA_Pred)

print(f"ARIMA MAE  : {ARIMA_MAE:.2f}")
print(f"ARMA RMSE  : {ARIMA_RMSE:.2f}")
print(f"ARIMA MAPE  :{ARIMA_MAPE:.2f}%")

XGB_MAE=mean_absolute_error(Y_test,XGB_Pred)
XGB_RMSE=root_mean_squared_error(Y_test,XGB_Pred)
XGB_MAPE=mean_absolute_percentage_error(Y_test,XGB_Pred)

print(f"XGB MAE   : {XGB_MAE:.2f}")
print(f"XGB RMSE  : {XGB_RMSE:.2f}")
print(f"XGB MAPE  : {XGB_MAPE:.2f}%")

LGBM_MAE=mean_absolute_error(Y_test,LGBM_Pred)
LGBM_RMSE=root_mean_squared_error(Y_test,LGBM_Pred)
LGBM_MAPE=mean_absolute_percentage_error(Y_test,LGBM_Pred)

print(f"LGBM MAE  : {LGBM_MAE:.2f}")
print(f"LGBM RMSE : {LGBM_RMSE:.2f}")
print(f"LGBM MAPE : {LGBM_MAPE:.2f}%")

LSTM_MAE=mean_absolute_error(Y_test_actual,LSTM_Pred)
LSTM_RMSE=root_mean_squared_error(Y_test_actual,LSTM_Pred)
LSTM_MAPE=mean_absolute_percentage_error(Y_test_actual,LSTM_Pred)

print(f"LSTM MAE  : {LSTM_MAE:.2f}")
print(f"LSTM RMSE : {LSTM_RMSE:.2f}")
print(f"LSTM MAPE : {LSTM_MAPE:.2f}%")

Comparision_Table=pd.DataFrame({
    "Models":["ARIMA","XGB","LGBM","LSTM"],
    "MAE":[ARIMA_MAE,XGB_MAE,LGBM_MAE,LSTM_MAE],
    "RMSE":[ARIMA_RMSE,XGB_RMSE,LGBM_RMSE,LSTM_RMSE],
    "MAPE":[ARIMA_MAPE,XGB_MAPE,LGBM_MAPE,LSTM_MAPE]
})
print(Comparision_Table)

plt.figure(figsize=(15,5))
plt.plot(train_st1.index,train_st1.values,label="Training",color="steelblue")
plt.plot(test_st1.index,test_st1.values,label="Actual",color="green")
plt.plot(test_st1.index,ARIMA_Pred.values,label="ARIMA Forecast",color="red",linestyle="--")
plt.xlabel("Date")
plt.ylabel("Weekly Sales")
plt.title("ARIMA Store-1 ACTUAL vs ARIMA Forecast")
plt.legend()
plt.tight_layout()
plt.show()


plt.figure(figsize=(15,5))
plt.plot(Y_test.values[:100],label="Actual",color="green")
plt.plot(XGB_Pred[:100],label="XGB-Prediction",color="orange",linestyle="--")
plt.xlabel("Sample Index")
plt.ylabel("Weekly_Sales")
plt.title("XGB - Prediction ACTUALL vs XGB")
plt.legend()
plt.tight_layout()
plt.show()


plt.figure(figsize=(15,5))
plt.plot(Y_test.values[:100],label="Actuall",color="green")
plt.plot(LGBM_Pred[:100],label="LGBM-Prediction",color="orange",linestyle="--")
plt.xlabel("Sample Index")
plt.ylabel("Weekly_Sales")
plt.title("LGBM - ACTUALL vs LGBM")
plt.legend()
plt.tight_layout()
plt.show()


plt.figure(figsize=(15,5))
plt.plot(Y_test_actual[:100],label="Actuall",color="green")
plt.plot(LSTM_Pred[:100],label="LSTM-Predition",color="purple",linestyle="--")
plt.xlabel("Sample Index")
plt.ylabel("Weekly Sales")
plt.title("LSTM - ACTUALL vs LSTM")
plt.legend()
plt.tight_layout()
plt.show()

import joblib
joblib.dump(LGBM_Model,"walmart_forecast_model.pkl")
print("Best model LGBM Saved Successfully")