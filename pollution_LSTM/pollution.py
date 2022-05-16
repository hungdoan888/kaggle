# -*- coding: utf-8 -*-
"""
Created on Mon May  9 21:23:06 2022

@author: hungd
https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
"""

#%% Libraries

from datetime import datetime
from math import sqrt

from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from numpy import concatenate
from matplotlib import pyplot

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

#%% For testing

inputFile = 'raw.csv'

#%% Load Data

def loadData(inputFile):
    # Read Data
    dataset = read_csv(inputFile,  parse_dates = [['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)
    
    # Drop No Column
    dataset.drop('No', axis=1, inplace=True)
    
    # manually specify column names
    dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
    dataset.index.name = 'date'
    
    # mark all NA values with 0
    dataset['pollution'].fillna(0, inplace=True)
    
    # drop the first 24 hours
    dataset = dataset[24:]
    return dataset

# Parse Date
def parse(x):
	return datetime.strptime(x, '%Y %m %d %H')


#%% Plot Data

def plotData(dataset):
    values = dataset.values
    # specify columns to plot
    groups = [0, 1, 2, 3, 5, 6, 7]
    i = 1
    # plot each column
    pyplot.figure()
    for group in groups:
    	pyplot.subplot(len(groups), 1, i)
    	pyplot.plot(values[:, group])
    	pyplot.title(dataset.columns[group], y=0.5, loc='right')
    	i += 1
    pyplot.show()
    
#%% Prepare Data for modeling

def prepareData(dataset):
    # Convert to values
    values = dataset.values
    
    # integer encode direction
    encoder = LabelEncoder()
    values[:,4] = encoder.fit_transform(values[:,4])
    
    # ensure all data is float
    values = values.astype('float32')
    
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    
    # frame as supervised learning
    reframed = series_to_supervised(scaled, 1, 1)
    
    # drop columns we don't want to predict
    reframed.drop(reframed.columns[[9,10,11,12,13,14,15]], axis=1, inplace=True)
    return reframed, scaler

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    # Define Variables
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
    
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
            
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
    
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

#%% Create X and y for train and test

def createXy(reframed):
    # split into train and test sets
    values = reframed.values
    n_train_hours = 365 * 24
    train = values[:n_train_hours, :]
    test = values[n_train_hours:, :]
    
    # split into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    return train_X, train_y, test_X, test_y

#%% Create and fit LSTM model

def createModel(train_X):
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    return model

def fitModel(model, train_X, train_y, test_X, test_y):
    history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
    
    # plot history
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()

#%% Evaluate Model

def evaluateModel(model, scaler, test_X, test_y):   
    # make a prediction
    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
    
    # invert scaling for forecast
    inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]
    
    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,0]
    
    # calculate RMSE
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)
    
#%% Main

if __name__ == "__main__":
    # Load Data
    dataset = loadData(inputFile)

    # Plot Data
    plotData(dataset)
    
    # Format data
    reframed, scaler = prepareData(dataset)
    
    # Create train test split with X y
    train_X, train_y, test_X, test_y = createXy(reframed)
    
    # Create LSTM model
    model = createModel(train_X)
    
    # Fit Model
    fitModel(model, train_X, train_y, test_X, test_y)
    
    # Evaluate Model
    evaluateModel(model, scaler, test_X, test_y)
