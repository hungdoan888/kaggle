# -*- coding: utf-8 -*-
"""
Created on Sat May  7 19:58:40 2022

@author: hungd
"""

#%% Libraries

import pandas as pd
from pandas.plotting import autocorrelation_plot
from pandas.tseries.offsets import DateOffset
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import statsmodels.api as sm
import statsmodels

#%% For testing

inputFile = './input/Perrin Freres monthly champagne sales millions.csv'

#%% Read Data

def readInputData(inputFile):
    # Read Data
    df = pd.read_csv(inputFile)
    
    # Change Column Names
    df.columns = ["Month", "Sales"]
    
    # Drop columns with nan
    df = df[~df['Sales'].isna()]

    # Convert Month to datetime
    df['Month'] = pd.to_datetime(df['Month'])
    
    # Set index
    df.set_index('Month',inplace=True)
    return df

#%% Describe and Plot data

def plotData(df):
    # Describe data
    df.describe()
    
    # Plot Data
    df.plot()

#%% Hypothesis Testing for stationary

def adfuller_test(sales):
    #HYPOTHESIS TEST:
    #Ho: It is non stationary
    #H1: It is stationary
    
    result=adfuller(sales)
    
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
    
    for value,label in zip(result,labels):
        print(label+' : '+str(value) )
    
    if result[1] <= 0.05:
        print(""""strong evidence against the null hypothesis(Ho), reject the null hypothesis. 
                  Data has no unit root and is stationary""")
    else:
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary")
        
#%% Difference the data to make it stationary

def differenceData(df):
    # Because there is a 1 year seasonality, it makes the most sense to difference the data with lags 12
    df['Seasonal First Difference']=df['Sales']-df['Sales'].shift(12)
    
    # Plot Data
    df['Seasonal First Difference'].plot()
    
    # Test to see if data is stationary
    adfuller_test(df['Seasonal First Difference'].dropna())
    return df

#%% Plot ACF and PACF

def findCorrelations(df):
    # Show Correlations
    autocorrelation_plot(df['Sales'])

    # Plot ACF and PACF
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(211)
    fig = plot_acf(df['Seasonal First Difference'].iloc[13:],lags=40,ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = plot_pacf(df['Seasonal First Difference'].iloc[13:],lags=40,ax=ax2)

#%% Create and fit model

def createFitARIMAModel(df):
    # Create Model
    model = statsmodels.tsa.arima.model.ARIMA(df['Sales'], order=(1,1,1))
    
    # Fit Model
    model_arima=model.fit()
    
    # Summarize Model
    model_arima.summary()    
    return model_arima
        
#%% Forecast ARIMA model

def forecastARIMA(model_arima, df):
    df['forecast_arima']=model_arima.predict(start=90,end=103,dynamic=True)
    df[['Sales','forecast_arima']].plot(figsize=(12,8))
    return df
    
#%% Create and fit SARIMA model

def createFitSARIMAModel(df):
    # Fit model
    model = sm.tsa.statespace.SARIMAX(df['Sales'],order=(1, 1, 1),seasonal_order=(1,1,1,12))
    
    # Fit Model
    model_sarima = model.fit()
    
    # Summarize Model
    model_sarima.summary()
    return model_sarima

#%% Forecast SARIMA

def forecastSARIMA(model_sarima, df):
    df['forecast_sarima'] = model_sarima.predict(start=90,end=103,dynamic=True)
    df[['Sales','forecast_sarima']].plot(figsize=(12,8))
    return df

#%% Predict on future dates

def predictOnFutureDates(model_sarima, df):
    # Create Future Dates
    future_dates=[df.index[-1]+ DateOffset(months=x)for x in range(0,24)]
    
    # Convert that list into DATAFRAME:
    future_datest_df = pd.DataFrame(index=future_dates[1:],columns=df.columns)
    
    # Merge with current df
    df_future = pd.concat([df,future_datest_df])
    
    #PREDICT
    df_future['forecast_future'] = model_sarima.predict(start = 104, end = 120, dynamic= True)  
    df_future[['Sales', 'forecast_future']].plot(figsize=(12, 8))
    return df_future

#%% Main

if __name__ == "__main__":
    # Read Data
    df = readInputData(inputFile)
    
    # Plot Data
    plotData(df)
    
    # Check if data is stationary
    adfuller_test(df['Sales'])
    
    # Difference Data
    df = differenceData(df)
    
    # Find Correlations
    findCorrelations(df)
    
    # Create and Fit ARIMA model
    model_arima = createFitARIMAModel(df)
    
    # Forecast ARIMA model
    df = forecastARIMA(model_arima, df)
    
    # Create and Fit SARIMA model
    model_sarima = createFitSARIMAModel(df)
    
    # Forecast SARIMA model
    df = forecastSARIMA(model_sarima, df)

    # Forecast Future
    df_future = predictOnFutureDates(model_sarima, df)