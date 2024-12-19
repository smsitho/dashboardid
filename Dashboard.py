import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np
from babel.numbers import format_currency
from matplotlib.ticker import ScalarFormatter
sns.set(style='dark')


def create_daily_df(df):
    daily_orders_df = df.resample(rule='D', on='dteday').agg({
        "instant": "nunique",
        "temp": "mean",
        "casual": "sum",
        "registered": "sum",
        "cnt": "sum"
    })
    daily_orders_df = daily_orders_df.reset_index()
    daily_orders_df.rename(columns={
        "dteday": "date",
        "cnt": "total"
    }, inplace=True)
    
    return daily_orders_df

def create_monthly_df(df):
    monthly_orders_df = df.resample(rule='M', on='dteday').agg({
        "instant": "nunique",
        "casual": "sum",
        "registered": "sum",
        "cnt": "sum"
    })
    monthly_orders_df.index = monthly_orders_df.index.strftime('%Y-%m')
    monthly_orders_df = monthly_orders_df.reset_index()
    monthly_orders_df.rename(columns={
        "dteday": "date",
        "cnt": "total"
    }, inplace=True)

    monthly_2011 = monthly_orders_df[monthly_orders_df['date'].str.contains('2011')]
    monthly_2012 = monthly_orders_df[monthly_orders_df['date'].str.contains('2012')]
    
    return monthly_orders_df, monthly_2011, monthly_2012
    
def create_sum_df(df):
    total_casual = df.casual.sum()
    total_registered = df.registered.sum()
    sum_df = np.array([total_casual, total_registered])
    return sum_df

def create_byseason_df(df):
    season_df = df.groupby(by="season").cnt.sum().reset_index()
    season_df["season"] = season_df["season"].map({
        1: "Spring",
        2: "Summer",
        3: "Fall",
        4: "Winter"
    })
    return season_df

def create_byhour_df(df):
    hourly = df.groupby(by="hr").agg({
        "instant": "nunique",
        "casual": sum ,
        "registered": sum,
        "cnt":  sum
    })
    return hourly

def create_bytime_df(df):
    time_bins = [0 ,5, 11, 15, 19,23]  
    time_labels = ["Malam","Pagi", "Siang", "Sore", "Malam"]
    
    hour_df['waktu'] = pd.cut(hour_df['hr'], bins=time_bins, labels=time_labels, right=False,ordered=False)
    time = hour_df.groupby(by="waktu").cnt.sum()
    time = pd.DataFrame(time)
    time = time.reset_index()
    time.rename(columns={
        "cnt": "Total Peminjaman"
    }, inplace=True)
    return time

def create_corr_df(df):
    corr_df = df.copy()
    corr_df = corr_df.drop(columns="dteday")
    return corr_df


hour_df = pd.read_csv("hour.csv")

datetime_columns = ["dteday"]
hour_df.sort_values(by="dteday", inplace=True)
hour_df.reset_index(inplace=True)
 
for column in datetime_columns:
    hour_df[column] = pd.to_datetime(hour_df[column])

min_date = hour_df["dteday"].min()
max_date = hour_df["dteday"].max()
 
with st.sidebar:
    
    start_date, end_date = st.date_input(
        label='Rentang Waktu',min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date]
    )

main_df = hour_df[(hour_df["dteday"] >= str(start_date)) & 
                (hour_df["dteday"] <= str(end_date))]

corr_df = create_corr_df(hour_df)
bytype_df = create_sum_df(hour_df)
season_df = create_byseason_df(hour_df)
daily_df = create_daily_df(hour_df)
hourly_total = create_byhour_df(hour_df)
daily_orders_df = create_daily_df(main_df)
time = create_bytime_df(hour_df)

monthly_orders_df, monthly_2011, monthly_2012 = create_monthly_df(main_df)
_, monthly2_2011 , monthly2_2012 = create_monthly_df(hour_df)
sum_df = create_sum_df(main_df)
hourly = create_byhour_df(main_df)


plt.rcParams.update({
    "axes.facecolor": "white",
    "figure.facecolor": "white"
})



st.header('Bike Sharing Dashboard :bike: :sparkles:')
tab1, tab2, = st.tabs(["Total", "Selected Period"])
 
with tab1:
    st.subheader('Summary of Total Rentals (2011-2012)')
    col1, col2 = st.columns([1,2])
    with col1:
        total_df = bytype_df[0]+bytype_df[1]
        st.metric("Total Rentals", value=total_df)
        st.metric("Average Daily Rentals", value=f"{daily_df['total'].mean():.2f}")
    with col2:
        
        col1, col2, = st.columns([1,1])
        with col1:
            casual_p = (bytype_df[0] / sum(bytype_df))*100
            registered_p = (bytype_df[1] / sum(bytype_df))*100
            st.metric("Casual customers", value=f"{casual_p:.2f}%")
            st.metric("Registered customers", value=f"{registered_p:.2f}%")
        with col2:
            fig, ax = plt.subplots(figsize=(2, 3))
            fig.patch.set_facecolor('white')
            plt.pie(bytype_df,labels = ["Casual","Registered"], textprops={'fontsize': 10})
            ax.set_title('Customer Type Demographics')
            st.pyplot(fig)


    
    
    st.subheader('Total Rentals per Month in 2011 and 2012')
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(30, 25))
    fig.patch.set_facecolor('white')
    
    ax[0].plot(monthly2_2011["date"], monthly2_2011["total"], marker='o', linewidth=2)  
    ax[0].plot(monthly2_2011["date"], monthly2_2011["casual"], marker='o', linewidth=2)  
    ax[0].plot(monthly2_2011["date"], monthly2_2011["registered"], marker='o', linewidth=2) 
    ax[0].set_title("Total Peminjaman per Bulan (2011)", loc="center", fontsize=40)
    ax[0].tick_params(axis='y', labelsize=35)
    ax[0].tick_params(axis='x', labelsize=30)
    ax[0].legend(['Total', 'Casual','Registered'],fontsize=30)
    
    
    ax[1].plot(monthly2_2012["date"], monthly2_2012["total"], marker='o', linewidth=2)  
    ax[1].plot(monthly2_2012["date"], monthly2_2012["casual"], marker='o', linewidth=2)  
    ax[1].plot(monthly2_2012["date"], monthly2_2012["registered"], marker='o', linewidth=2)  
    ax[1].set_title("Total Peminjaman per Bulan (2012)", loc="center", fontsize=40)
    ax[1].tick_params(axis='y', labelsize=35)
    ax[1].tick_params(axis='x', labelsize=30)
    ax[1].legend(['Total', 'Casual','Registered'],fontsize=30)
     
    st.pyplot(fig)
    
    
    st.subheader('Correlation Between Variables')
    fig, ax = plt.subplots(figsize=(20, 10))
    sns.heatmap(corr_df.corr()) 
    ax.tick_params(axis='y', labelsize=20)
    ax.tick_params(axis='x', labelsize=15)
    st.pyplot(fig)
    
    st.subheader('Total Rentals by Season')
    col1, col2, col3,col4 = st.columns([1,1,1,1])
    
    with col1:
        st.metric("Total rental in spring", value=season_df['cnt'][0])
        
    with col2:
        st.metric("Total rental in summer", value=season_df['cnt'][1])
                  
    with col3:
        st.metric("Total rental in fall", value=season_df['cnt'][2])
    with col4:
        st.metric("Total rental in winter", value=season_df['cnt'][3])
    
                  
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(40, 15))
    sns.barplot(
        x="season",
        y="cnt",
        data=season_df,
    )
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    formatter.set_useOffset(False)
    ax.yaxis.set_major_formatter(formatter)
    ax.set_xlabel(None)
    ax.tick_params(axis='x', labelsize=40)
    ax.tick_params(axis='y', labelsize=40)
    st.pyplot(fig)

    st.subheader('Total Rentals per Hour')

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(30, 15))
    ax.plot(hourly_total.index.get_level_values('hr'), hourly_total["cnt"], marker='o', linewidth=2)
    ax.plot(hourly_total.index.get_level_values('hr'), hourly_total["casual"], marker='o', linewidth=2)
    ax.plot(hourly_total.index.get_level_values('hr'), hourly_total["registered"], marker='o', linewidth=2)
    ax.set_xticks(hourly_total.index.get_level_values('hr'))
    ax.tick_params(labelsize=35)
    ax.tick_params(labelsize=35)
    ax.legend(['Total', 'Casual','Registered'],fontsize=30)
    st.pyplot(fig)

    st.subheader('Total Rentals per Time Periods of the Day')
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 15))
    sns.barplot(
        x="waktu",
        y="Total Peminjaman",
        data=time,
    )
    
    ax.set_ylabel(None)
    ax.set_xlabel(None)
    ax.yaxis.set_major_formatter(formatter)
    ax.tick_params(axis='x', labelsize=30)
    ax.tick_params(axis='y', labelsize=20)
    st.pyplot(fig)

with tab2:
    st.subheader(f"Rentals Data Summary from {start_date} to {end_date}")
     
    col1, col2 = st.columns([1.5,4])
     
    with col1:
        
        total_orders = daily_orders_df.total.sum()
        st.metric("Total rental", value=total_orders)
        st.metric("Average Daily Rentals", value=f"{daily_orders_df['total'].mean():.2f}")
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.pie(sum_df,labels = ["Casual","Registered"],autopct='%1.0f%%',textprops={'fontsize': 25})
        st.pyplot(fig)
     
    with col2:
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.plot(
            daily_orders_df["date"],
            daily_orders_df["total"],
            marker='o', 
            linewidth=2,
            color="#90CAF9"
        )
        ax.set_title('Daily Total Rentals',fontsize=20)
        ax.tick_params(axis='y', labelsize=20)
        ax.tick_params(axis='x', labelsize=15)
         
        st.pyplot(fig)

    st.subheader('Total Rentals per Hour')

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(30, 15))
    ax.plot(hourly.index.get_level_values('hr'), hourly["cnt"], marker='o', linewidth=2)
    ax.plot(hourly.index.get_level_values('hr'), hourly["casual"], marker='o', linewidth=2)
    ax.plot(hourly.index.get_level_values('hr'), hourly["registered"], marker='o', linewidth=2)
    ax.set_xticks(hourly.index.get_level_values('hr'))
    ax.tick_params(labelsize=35)
    ax.tick_params(labelsize=35)
    ax.legend(['Total', 'Casual','Registered'],fontsize=30)
    st.pyplot(fig)
    
    st.subheader("Total Rentals per month")
    
    if any(monthly_orders_df['date'].str.contains('2011')) and any(monthly_orders_df['date'].str.contains('2012')):
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(35, 25))
        ax[0].plot(monthly_2011["date"], monthly_2011["total"], marker='o', linewidth=2)  
        ax[0].plot(monthly_2011["date"], monthly_2011["casual"], marker='o', linewidth=2)  
        ax[0].plot(monthly_2011["date"], monthly_2011["registered"], marker='o', linewidth=2) 
        ax[0].set_title("Total Peminjaman per Bulan (2011)", loc="center", fontsize=20)
        ax[0].tick_params(axis='y', labelsize=35)
        ax[0].tick_params(axis='x', labelsize=30)
        ax[0].legend(['Total', 'Casual','Registered'],fontsize=30)
        
        ax[1].plot(monthly_2012["date"], monthly_2012["total"], marker='o', linewidth=2)  
        ax[1].plot(monthly_2012["date"], monthly_2012["casual"], marker='o', linewidth=2)  
        ax[1].plot(monthly_2012["date"], monthly_2012["registered"], marker='o', linewidth=2)  
        ax[1].set_title("Total Peminjaman per Bulan (2012)", loc="center", fontsize=40)
        ax[1].tick_params(axis='y', labelsize=35)
        ax[1].tick_params(axis='x', labelsize=30)
        ax[1].legend(['Total', 'Casual','Registered'],fontsize=30)
        st.pyplot(fig)
    
    elif all(monthly_orders_df['date'].str.contains('2011')) :
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(35, 15))
        ax.plot(monthly_2011["date"], monthly_2011["total"], marker='o', linewidth=2)  
        ax.plot(monthly_2011["date"], monthly_2011["casual"], marker='o', linewidth=2)  
        ax.plot(monthly_2011["date"], monthly_2011["registered"], marker='o', linewidth=2) 
        ax.set_title("Total Peminjaman per Bulan (2011)", loc="center", fontsize=20)
        ax.tick_params(axis='y', labelsize=35)
        ax.tick_params(axis='x', labelsize=30)
        ax.legend(['Total', 'Casual','Registered'],fontsize=30)
        st.pyplot(fig)
    
    else :
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(35, 15))
        ax.plot(monthly_2012["date"], monthly_2012["total"], marker='o', linewidth=2)  
        ax.plot(monthly_2012["date"], monthly_2012["casual"], marker='o', linewidth=2)  
        ax.plot(monthly_2012["date"], monthly_2012["registered"], marker='o', linewidth=2)  
        ax.set_title("Total Peminjaman per Bulan (2012)", loc="center", fontsize=40)
        ax.tick_params(axis='y', labelsize=35)
        ax.tick_params(axis='x', labelsize=30)
        ax.legend(['Total', 'Casual','Registered'],fontsize=30)
        st.pyplot(fig)



 
st.caption('Copyright (c) smsitho 2024 | [Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/bike-sharing-dataset/data)')
