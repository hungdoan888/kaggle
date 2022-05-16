# -*- coding: utf-8 -*-

#%% Libraries

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score

#%% For Testing

inputFile = r'burke-gilman-trail-north-of-ne-70th-st-bike-and-ped-counter.csv'
SEED = 42  # For model
N = 5  # Cross validation

#%% Import Data

def importData(inputFile):
    # Read File
    df = pd.read_csv(inputFile)

    # Rename Columns
    df = df.rename(columns={'Date': 'date',
                            'BGT North of NE 70th Total': 'total',
                            'Ped South': 'pedSouth',
                            'Ped North': 'pedNorth',
                            'Bike North': 'bikeNorth',
                            'Bike South': 'bikeSouth'})
    
    # Change date to standard date
    df['date'] = df['date'].apply(lambda x: pd.to_datetime(x))
    
    # Sort by date
    df = df.sort_values('date')
       
    # Separate Date
    df.insert(loc=1, column='year', value=df['date'].apply(lambda x: x.year))
    df.insert(loc=2, column='month', value=df['date'].apply(lambda x: x.month))
    df.insert(loc=3, column='day', value=df['date'].apply(lambda x: x.day))
    df.insert(loc=4, column='hour', value=df['date'].apply(lambda x: x.hour))
    
    # Create north and south traffic
    df['north'] = df['pedNorth'] + df['bikeNorth']
    df['south'] = df['pedSouth'] + df['bikeSouth']
    df['greaterNorthTraffic'] = df.apply(lambda row: 1 if row['north'] >= row['south'] else 0, axis=1)
    
    # Anomaly
    df['anomaly'] = df['total'].apply(lambda x: 1 if x >= 500 else 0)
    return df

#%% Get dataframe with NaNs

def getAllNaNs(df):
    df_nan = df[df['total'].isna()]
    return df_nan

#%% Create Traffic versus Time Plot

def trafficTimePlot(df):
    # Total traffic vs time
    plt.plot(df['date'], df['total'])
    plt.axhline(y=500, color='r', linestyle='-')
    plt.grid(color='grey', linestyle='-', linewidth=.5)
    plt.xlabel("Time")
    plt.ylabel("Total Traffic")
    plt.title ('Total Traffic vs Time')
    
#%% Plot histograms

def plotHistograms(df):
    # By year
    df_year = df.groupby('year')['total'].mean().to_frame('yearAvg').reset_index()
    df_year.plot.bar(x='year', y='yearAvg')
    
    # By month
    df_month = df.groupby('month')['total'].mean().to_frame('monthAvg').reset_index()
    df_month.plot.bar(x='month', y='monthAvg')
    
    # By day
    df_day = df.groupby('day')['total'].mean().to_frame('dayAvg').reset_index()
    df_day.plot.bar(x='day', y='dayAvg')
    
    # By hour
    df_hour = df.groupby('hour')['total'].mean().to_frame('hourAvg').reset_index()
    df_hour.plot.bar(x='hour', y='hourAvg')
  
#%% Plot Month vs hour

def monthHourPlot(df):
    # Number of anomalies by month hour
    df_numAnom = df.groupby(['month', 'hour'])['anomaly'].sum().reset_index()
    df_numAnom = df_numAnom.pivot('hour', 'month', 'anomaly')
    ax1 = sns.heatmap(df_numAnom, annot=True, cmap='Blues', fmt="d")    
    ax1.set_title('Number of Anomalies by month and hour')
    
    # Mean total by month hour
    df_meanTotal = df.groupby(['month', 'hour'])['total'].mean().reset_index()
    df_meanTotal = df_meanTotal.pivot('hour', 'month', 'total')
    ax1 = sns.heatmap(df_meanTotal, cmap='Blues', fmt="d") 
    
    # Binary plots of month hour by failures
    df_anom = df[df['anomaly'] == 1]
    df_norm = df[df['anomaly'] == 0]
    plt.scatter(df_norm['month'], df_norm['hour'], c='b', s=2)
    plt.scatter(df_anom['month'], df_anom['hour'], c='r', s=2)
    plt.xlabel("Month")
    plt.ylabel("Hour")
    plt.title ('Number of Anomalies by month and hour')
        
#%% Outlier Detection using dbscan (Unsupervised method for outlier detection)

def outlierDetection(df):
    # Get min outlier threshold 
    minOutlierValue = useDBSCAN(df)
    
    # Change values of pedSouth
    df['pedSouth'] = df.apply(lambda row: row['pedSouth'] if row['total'] < minOutlierValue 
                                                          else row['pedSouth'] * minOutlierValue // row['total'], 
                                                          axis=1)
    
    # Change values of pedNorth
    df['pedNorth'] = df.apply(lambda row: row['pedNorth'] if row['total'] < minOutlierValue 
                                                          else row['pedNorth'] * minOutlierValue // row['total'], 
                                                          axis=1)
    
    # Change values of bikeSouth
    df['bikeSouth'] = df.apply(lambda row: row['bikeSouth'] if row['total'] < minOutlierValue 
                                                          else row['bikeSouth'] * minOutlierValue // row['total'], 
                                                          axis=1)
    
    # Change values of bikeNorth
    df['bikeNorth'] = df.apply(lambda row: row['bikeNorth'] if row['total'] < minOutlierValue 
                                                          else row['bikeNorth'] * minOutlierValue // row['total'], 
                                                          axis=1)
    
    # Recalculate the total
    df['total'] = df['pedSouth'] + df['pedNorth'] + df['bikeSouth'] + df['bikeNorth']
    return df
    
# Density Based Spacial Clustering 
def useDBSCAN(df):
    # Convert to np array
    dbValue = df['total'].to_numpy()
    dbDate = df['date'].to_numpy()
    
    # Get rid of NaNs
    notNanIdx = ~np.isnan(dbValue)
    dbValue = dbValue[notNanIdx].reshape(-1,1)
    dbDate = dbDate[notNanIdx].reshape(-1,1)
    
    # Find outliers
    clusters = DBSCAN(eps=20, min_samples=2).fit(dbValue).labels_
    
    # Plot Results
    plt.scatter(dbDate, dbValue, c=clusters)
    
    # Take the group with the highest values
    outliers = dbValue[clusters==-1]
    
    # Min outlier value
    minOutlierValue = int(outliers.min())
    return minOutlierValue

#%% Impute missing values

def imputation(df):
    # Create a copy of df to manipulate and append results to df
    df_imp = df.copy(deep=True)
    
    # Cumsum
    df_imp['totalCumsum'] = df_imp.groupby(['month', 'day', 'hour'])['total'].cumsum()
    df_imp['pedSouthCumsum'] = df_imp.groupby(['month', 'day', 'hour'])['pedSouth'].cumsum()
    df_imp['pedNorthCumsum'] = df_imp.groupby(['month', 'day', 'hour'])['pedNorth'].cumsum()
    df_imp['bikeSouthCumsum'] = df_imp.groupby(['month', 'day', 'hour'])['bikeSouth'].cumsum()
    df_imp['bikeNorthCumsum'] = df_imp.groupby(['month', 'day', 'hour'])['bikeNorth'].cumsum()

    # Rank
    df_imp['rank'] = df_imp.groupby(['month', 'day', 'hour'])['year'].rank()
    
    # CumMean
    df_imp['totalCumMean'] = df_imp['totalCumsum'] / df_imp['rank']
    df_imp['pedSouthCumMean'] = df_imp['pedSouthCumsum'] / df_imp['rank']
    df_imp['pedNorthCumMean'] = df_imp['pedNorthCumsum'] / df_imp['rank']
    df_imp['bikeSouthCumMean'] = df_imp['bikeSouthCumsum'] / df_imp['rank']
    df_imp['bikeNorthCumMean'] = df_imp['bikeNorthCumsum'] / df_imp['rank']
    
    # Sort Values by month, day, hour
    df = df.sort_values(['month', 'day', 'hour'])
    df_imp = df_imp.sort_values(['month', 'day', 'hour'])
    
    # Add Total Cumulative Mean to df
    df['totalCumMean'] = df_imp['totalCumMean']
    
    # Impute values
    df = df.apply(lambda row: insertImputedValues(row, df, df_imp), axis=1)
    
    # Sort Values by time
    df = df.sort_values('date')
    return df
     
def insertImputedValues(row, df, df_imp):
    # Define variables
    year = row['year']
    month = row['month']
    day = row['day']
    hour = row['hour']
    total = row['total']
    
    # Return current values if nothing to impute
    if not pd.isnull(total):
        return row
    
    # If year is 2014, return previous hours value
    if year == 2014:
        # Take previous hour sample
        df_temp = df[(df['year'] == year) & (df['month'] == month) & (df['day'] == day) & (df['hour'] == hour-1)]
        
        row['total'] = df_temp['total'].iloc[0]                # total
        row['totalCumMean'] = df_temp['totalCumMean'].iloc[0]  # totalCumMean
        row['pedSouth'] = df_temp['pedSouth'].iloc[0]          # pedSouth
        row['pedNorth'] = df_temp['pedNorth'].iloc[0]          # pedNorth
        row['bikeSouth'] = df_temp['bikeSouth'].iloc[0]        # bikeSouth
        row['bikeNorth'] = df_temp['bikeNorth'].iloc[0]        # bikeNorth
        return row
    
    # Take previous year's value
    df_impTemp = df_imp[(df_imp['year'] == year-1) & (df_imp['month'] == month) & 
                        (df_imp['day'] == day) & (df_imp['hour'] == hour)]
      
    row['total'] = df_impTemp['totalCumMean'].iloc[0]          # total
    row['totalCumMean'] = df_impTemp['totalCumMean'].iloc[0]   # totalCumMean
    row['pedSouth'] = df_impTemp['pedSouthCumMean'].iloc[0]    # pedSouth
    row['pedNorth'] = df_impTemp['pedNorthCumMean'].iloc[0]    # pedNorth
    row['bikeSouth'] = df_impTemp['bikeSouthCumMean'].iloc[0]  # bikeSouth
    row['bikeNorth'] = df_impTemp['bikeNorthCumMean'].iloc[0]  # bikeNorth
    return row

#%% Get Month Hour cumulative mean

def monthHourCumMean(df):
    # Create a copy of df to manipulate
    df_monthHour = df.copy(deep=True)
    
    # Sort df and df_monthHour
    df = df.sort_values(['month', 'hour'])
    df_monthHour = df_monthHour.sort_values(['month', 'hour'])
    
    # Group by month hour and get cumulative sum of anomaly and total
    df_monthHour['totalCumSum'] = df_monthHour.groupby(['month', 'hour'])['total'].cumsum()
    df_monthHour['anomalyCumSum'] = df_monthHour.groupby(['month', 'hour'])['anomaly'].cumsum()
    
    # Get rank to calculate cumulative means
    df_monthHour['rank'] = df_monthHour.groupby(['month', 'hour'])['year'].rank(method='first')
    
    # Calculate cumulative mean for anomaly and total
    df['monthHourTotalCumMean'] = df_monthHour['totalCumSum'] / df_monthHour['rank']
    df['monthHourAnomalyCumMean'] = df_monthHour['anomalyCumSum'] / df_monthHour['rank']
    
    # Sort df by time
    df = df.sort_values('date')
    return df
    
#%% Lag total by 3 and calculate rate of change

def getLags(df):
    # Lag total by 1
    df['totalLag1'] = df['total'].shift()
    
    # Get rate of change since last 1
    df['totalRate1'] = df['total'] - df['totalLag1']
    
    # Anomoly lag1
    df['anomalyLag1'] = df['anomaly'].shift()
    
    # Get total future values to use for prediction
    df['anomalyFuture'] = df['anomaly'].shift(-3)
    df['totalFuture'] = df['total'].shift(-3)
    
    # Drop first row and last 3 due to creating nans
    df = df.dropna()
    return df

#%% Transform time

def transformTime(df):
    df = sinCosEncoding(df, 'month')  # month
    df = sinCosEncoding(df, 'day')    # day
    df = sinCosEncoding(df, 'hour')   # hour
    return df

# Sin Cos Encoding for time
def sinCosEncoding(df, column):
    max_value = df[column].max()
    df[column + '_cos'] = df[column].apply(lambda x: math.cos((2 * math.pi * x) / max_value))
    df[column + '_sin'] = df[column].apply(lambda x: math.sin((2 * math.pi * x) / max_value))
    return df

#%% Reduce df down to only columns for modeling

def getColumnsForModel(df):
    df_model = df[['month_cos', 'month_sin', 
                   'day_cos', 'day_sin',
                   'hour_cos', 'hour_sin', 
                   'total', 'totalCumMean', 'totalLag1', 'totalRate1',
                   'monthHourTotalCumMean',
                   'anomaly', 'anomalyLag1',
                   'monthHourAnomalyCumMean',
                   'anomalyFuture']]
    return df_model

#%% Apply standard scaling to numeric features

def scaleFeatures(df_model):
    # Create a df of numeric features to scale
    df_num = df_model[['total', 'totalCumMean', 'totalLag1', 'totalRate1', 
                       'monthHourTotalCumMean','monthHourAnomalyCumMean']]
    
    # Drop numeric features from df_model
    df_model = df_model.drop(['total', 'totalCumMean', 'totalLag1', 'totalRate1', 
                              'monthHourTotalCumMean','monthHourAnomalyCumMean'], axis=1)
    
    # Scale numeric features
    scaledFeatures = StandardScaler().fit_transform(df_num)
    df_scaled = pd.DataFrame(scaledFeatures, index=df_num.index, columns=df_num.columns)
    
    # Horizontally concatenate 
    df_model = pd.concat([df_model, df_scaled], axis=1)

    # Reorganize order
    df_model = getColumnsForModel(df_model)
    return df_model

#%% Use a standard scalar to scale variables

def defineXandy(df_model):
    # Use last 27% for testing.  This gives 110 anomalies in test set and 386 in training
    testStartIdx = getTestStartIdx(df_model)
    
    # Train and test
    df_train = df_model.iloc[:testStartIdx]
    df_test = df_model.iloc[testStartIdx:]
    
    # Define X
    X_train = df_train.drop(columns='anomalyFuture').to_numpy()
    X_test = df_test.drop(columns='anomalyFuture').to_numpy()

    # Define y
    y_train = df_train['anomalyFuture'].values
    y_test = df_test['anomalyFuture'].values
    return df_train, df_test, X_train, X_test, y_train, y_test

# Get testStartIdx
def getTestStartIdx(df_model):
    # Use last 27% for testing.  This gives 110 anomalies in test set and 386 in training
    testStartIdx = int(len(df_model) - (len(df_model) * .27))
    return testStartIdx

#%% Create model

def createModel(SEED):
    model = RandomForestClassifier(criterion='gini',
                                   n_estimators=1750,
                                   max_depth=7,
                                   min_samples_split=6,
                                   min_samples_leaf=6,
                                   max_features='auto',
                                   oob_score=True,
                                   random_state=SEED,
                                   n_jobs=-1,
                                   verbose=1)
    return model

#%% Evaluate Model

def evaluateModel(model, df_train, X_train, y_train, X_test, N):
    # Out of bag score to predict accuracy of testing set
    oob = 0
    
    # Give probability of survival
    df_pred = pd.DataFrame(np.zeros((len(X_test), N * 2)), 
                           columns=['Fold_{}_Prob_{}'.format(i, j) for i in range(1, N + 1) for j in range(2)])
    
    # Dataframe to keep track of fprs, tprs, trnAucScore, and valAucScore
    df_roc = pd.DataFrame()
    
    # Feature Importance
    columns = list(df_train.columns)
    columns.remove('anomalyFuture')
    df_featImp = pd.DataFrame(np.zeros((X_train.shape[1], N)), 
                              columns=['Fold_{}'.format(i) for i in range(1, N + 1)], 
                              index=columns)
    
    # Stratified crossfold validaiton
    skf = StratifiedKFold(n_splits=N, random_state=N, shuffle=True)
    for fold, (trnIdx, valIdx) in enumerate(skf.split(X_train, y_train), 1):
        print('Fold {}\n'.format(fold))
        
        # Fitting the model
        model.fit(X_train[trnIdx], y_train[trnIdx])
        
        # Compute oob score
        oob += model.oob_score_ / N
        print('Fold {} OOB Score: {}\n'.format(fold, model.oob_score_))   
        
        # Computing AUC score
        df_roc = getROCCurveValues(model, X_train, y_train, trnIdx, valIdx, df_roc)
        
        # Get Prediction Probabilities
        df_pred = getPredProb(model, fold, df_pred, X_test)
        
        # Feature Importance
        df_featImp = getFeatImp(model, df_featImp, fold)
        
    print('Average OOB Score: {}'.format(oob))
    return df_pred, df_roc, df_featImp, oob

# ROC Curve Values
def getROCCurveValues(model, X_train, y_train, trnIdx, valIdx, df_roc):    
    # Train
    trnFpr, trnTpr, trnThresholds = roc_curve(y_train[trnIdx], model.predict_proba(X_train[trnIdx])[:, 1])
    trnAucScore = auc(trnFpr, trnTpr)
    
    # Validation
    valFpr, valTpr, valThresholds = roc_curve(y_train[valIdx], model.predict_proba(X_train[valIdx])[:, 1])
    valAucScore = auc(valFpr, valTpr)
    
    # Append Scores
    df_rocTemp = pd.DataFrame({'fprs': [valFpr], 
                               'tprs': [valTpr], 
                               'trnAucScore': [trnAucScore], 
                               'valAucScore': [valAucScore]})
    df_roc = pd.concat([df_roc, df_rocTemp])
    return df_roc

# Get Prediction Probabilities
def getPredProb(model, fold, df_pred, X_test):
    # X_test probabilities
    df_pred.loc[:, 'Fold_{}_Prob_0'.format(fold)] = model.predict_proba(X_test)[:, 0]
    df_pred.loc[:, 'Fold_{}_Prob_1'.format(fold)] = model.predict_proba(X_test)[:, 1]
    return df_pred

# Feature Importance
def getFeatImp(model, df_featImp, fold):
    df_featImp.iloc[:, fold - 1] = model.feature_importances_
    return df_featImp

#%% Plot Feature Importance

def plotFeatImp(df_featImp):
    df_featImp['Mean_Importance'] = df_featImp.mean(axis=1)
    df_featImp.sort_values(by='Mean_Importance', inplace=True, ascending=False)
    
    plt.figure(figsize=(15, 20))
    sns.barplot(x='Mean_Importance', y=df_featImp.index, data=df_featImp)
    
    plt.xlabel('')
    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)
    plt.title('Random Forest Classifier Mean Feature Importance Between Folds', size=15)
    
    plt.show()
    
#%% Plot ROC Curve

def plot_roc_curve(df_roc):
    # Define fprs and tprs
    fprs = list(df_roc['fprs'])
    tprs = list(df_roc['tprs'])
    
    tprs_interp = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    f, ax = plt.subplots(figsize=(15, 15))
    
    # Plotting ROC for each fold and computing AUC scores
    for i, (fpr, tpr) in enumerate(zip(fprs, tprs), 1):
        tprs_interp.append(np.interp(mean_fpr, fpr, tpr))
        tprs_interp[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        ax.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC Fold {} (AUC = {:.3f})'.format(i, roc_auc))
        
    # Plotting ROC for random guessing
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=0.8, label='Random Guessing')
    
    mean_tpr = np.mean(tprs_interp, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    
    # Plotting the mean ROC
    ax.plot(mean_fpr, mean_tpr, color='b', label='Mean ROC (AUC = {:.3f} $\pm$ {:.3f})'.format(mean_auc, std_auc), 
            lw=2, alpha=0.8)
    
    # Plotting the standard deviation around the mean ROC Curve
    std_tpr = np.std(tprs_interp, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label='$\pm$ 1 std. dev.')
    
    ax.set_xlabel('False Positive Rate', size=15, labelpad=20)
    ax.set_ylabel('True Positive Rate', size=15, labelpad=20)
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])

    ax.set_title('ROC Curves of Folds', size=20, y=1.02)
    ax.legend(loc='lower right', prop={'size': 13})
    
    plt.show()
    
#%% Make Prediction

def makePrediction(model, df_pred, df, y_test, N):
    # Class anomaly are columns that end with Prob_1
    class_anomaly = [col for col in df_pred.columns if col.endswith('Prob_1')]
    
    # Average anomaly and not anomaly classes
    df_pred['1'] = df_pred[class_anomaly].sum(axis=1) / N
    df_pred['0'] = df_pred.drop(columns=class_anomaly).sum(axis=1) / N
    
    # Initiate prediction column
    df_pred['pred'] = 0
    
    # Get indices of anomaly and set pred = 1 if anomaly
    pos = df_pred[df_pred['1'] >= 0.5].index
    df_pred.loc[pos, 'pred'] = 1
    
    # Predictions
    y_pred = df_pred['pred'].astype(int)
    
    # Get test set
    testStartIdx = getTestStartIdx(df)
    df_test = df[testStartIdx:]
    
    # Create submissions df with date and anomaly
    df_submission = pd.DataFrame()
    df_submission['date'] = df_test['date']
    df_submission['Predicted'] = y_pred.values
    df_submission['Actual'] = y_test
    
    # Count how many correct
    df_submission['Correct'] = df_submission.apply(lambda row: 1 if row['Predicted'] == row['Actual'] else 0, axis=1)
    
    # Ensure submission not based on imputed anomolies
    df_submission = df_submission[~df_submission['date'].isin(list(df_nan['date']))]
    
    # Calculate Metrics
    accuracy = accuracy_score(df_submission['Actual'], df_submission['Predicted'])    # Accuracy
    precision = precision_score(df_submission['Actual'], df_submission['Predicted'])  # precision
    recall = recall_score(df_submission['Actual'], df_submission['Predicted'])        # recall
    f1 = f1_score(df_submission['Actual'], df_submission['Predicted'])                # f1
    return df_submission, accuracy, precision, recall, f1

#%% Main

if __name__ == '__main__':
    # Import Data
    df = importData(inputFile)
    
    # Get df with NaNs
    df_nan = getAllNaNs(df)
    
    # Create plot for date vs total traffic
    trafficTimePlot(df)
    
    # Plot Histograms for year, month, day, hour
    plotHistograms(df)
    
    # Create heatmap of month vs hour
    monthHourPlot(df)
    
    # Outlier Detection
    df = outlierDetection(df)
    trafficTimePlot(df)
    
    # Imputation
    df = imputation(df)
    trafficTimePlot(df)
    
    # Get Cumulative means for month hour
    df = monthHourCumMean(df)
    
    # Get lags
    df = getLags(df)
    
    # Transform time
    df = transformTime(df)
    
    # Reduce df down to only columns for modeling
    df_model = getColumnsForModel(df)
    
    # Scale numeric features
    df_model = scaleFeatures(df_model)
    
    # Define X and y
    df_train, df_test, X_train, X_test, y_train, y_test = defineXandy(df_model)
    
    # Create Model
    model = createModel(SEED)

    # Evaluate Model
    df_pred, df_roc, df_featImp, oob = evaluateModel(model, df_train, X_train, y_train, X_test, N)
    
    # Plot Feature Importance
    plotFeatImp(df_featImp)
    
    # Plot ROC curve
    plot_roc_curve(df_roc)
    
    # Make Prediction
    df_submission, accuracy, precision, recall, f1 = makePrediction(model, df_pred, df, y_test, N)
