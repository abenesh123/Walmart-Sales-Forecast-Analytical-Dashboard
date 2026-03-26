import pandas as pd
import mysql.connector
from sqlalchemy import create_engine

df=pd.read_csv("D:\DS-projects\Time Series Forecasting\walmart-sales-dataset-of-45stores.csv")

df["Date"]=pd.to_datetime(df["Date"],format="%d-%m-%Y")

engine = create_engine("mysql+mysqlconnector://root:abineshmysql%40123@localhost/forecasting")

df.to_sql("sales",con=engine,if_exists="replace",index=False)
print("Data loaded successfully")
print(f"total rows :{len(df)}")