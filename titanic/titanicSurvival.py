# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 12:12:26 2022

@author: hungd
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%% Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold

import string
import warnings

#%% For Testing

inputTrain = r'C:\Users\hungd\Desktop\python_Practice\titanic\input\train.csv'
inputTestX = r'C:\Users\hungd\Desktop\python_Practice\titanic\input\test.csv'
inputTesty = r'C:\Users\hungd\Desktop\python_Practice\titanic\input\gender_submission.csv'

# Define global variables
SEED = 42
N = 5

#%% Suppress Warnings

warnings.filterwarnings('ignore')

#%% Seaborn Settings

sns.set(style="darkgrid")

#%% Import Data

def importData(inputTrain, inputTestX, inputTesty):
    # Training Data
    df_train = pd.read_csv(inputTrain)

    # Test Data
    X_test = pd.read_csv(inputTestX)
    y_test = pd.read_csv(inputTesty)
    y_test = y_test[['Survived']]
    df_test = pd.concat([X_test, y_test], 1)
    
    # Create a dataset with all data
    df = pd.concat([df_train, df_test]).reset_index(drop=True)
    return df
 
#%% Determine which feature Age is most closely correlated to

def corrAge(df):
    df_corr = df.corr().abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()
    df_corr.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)
    df_corrAge = df_corr[df_corr['Feature 1'] == 'Age']
    print(df_corrAge)
    
#%% Impute Age by Passenger class and sex

def imputeAge(df):
    # Get median age by sex and class to use for imputation
    df_ageByClassSex = df.groupby(['Sex', 'Pclass'])['Age'].median()
    print(df_ageByClassSex)
    
    # Filling the missing values in Age with the medians of Sex and Pclass groups
    df['Age'] = df.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))
    return df

#%% Impute Embark

def imputeEmbark(df):
    # It is known from an outside resource that the two missing values from embark are 'S'
    df['Embarked'] = df['Embarked'].fillna('S')
    return df

#%% Impute Fare

def imputeFare(df):
    # Filling the missing value in Fare with the median Fare of 3rd class alone passenger
    med_fare = df.groupby(['Pclass', 'Parch', 'SibSp'])['Fare'].median()[3][0][0]
    df['Fare'] = df['Fare'].fillna(med_fare)
    return df

#%% Deck Imputation

def imputeDeck(df):
    # Creating Deck column from the first letter of the Cabin column (M stands for Missing)
    df['Deck'] = df['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')
    
    # Passenger in the T deck is changed to A
    idx = df[df['Deck'] == 'T'].index
    df.loc[idx, 'Deck'] = 'A'
    
    # Group decks
    df['Deck'] = df['Deck'].replace(['A', 'B', 'C'], 'ABC')
    df['Deck'] = df['Deck'].replace(['D', 'E'], 'DE')
    df['Deck'] = df['Deck'].replace(['F', 'G'], 'FG')
    
    # Pring Counts
    print(df['Deck'].value_counts())
    
    # Drop Cabin and use deck instead
    df = df.drop('Cabin', 1)
    return df
    
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

#%% Divide df

def divide_df(df):
    # Returns divided dfs of training and test set
    return df.loc[:890], df.loc[891:].drop(['Survived'], axis=1)

#%% Evaulaute Model

def evaluateModel(model, X_train, y_train, X_test, y_test, N):
    # Out of bag score to predict accuracy of testing set
    oob = 0
    
    # Give probability of survival
    probs = pd.DataFrame(np.zeros((len(X_test), N * 2)), 
                         columns=['Fold_{}_Prob_{}'.format(i, j) for i in range(1, N + 1) for j in range(2)])
    
    # Empty array to keep track of scores
    fprs, tprs, scores = [], [], []
    
    # Stratified crossfold validaiton
    skf = StratifiedKFold(n_splits=N, random_state=N, shuffle=True)
    for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        print('Fold {}\n'.format(fold))
        
        # Fitting the model
        model.fit(X_train[trn_idx], y_train[trn_idx])
        
        # Computing Train AUC score
        trn_fpr, trn_tpr, trn_thresholds = roc_curve(y_train[trn_idx], model.predict_proba(X_train[trn_idx])[:, 1])
        trn_auc_score = auc(trn_fpr, trn_tpr)
        
        # Computing Validation AUC score
        val_fpr, val_tpr, val_thresholds = roc_curve(y_train[val_idx], model.predict_proba(X_train[val_idx])[:, 1])
        val_auc_score = auc(val_fpr, val_tpr)  
          
        scores.append((trn_auc_score, val_auc_score))
        fprs.append(val_fpr)
        tprs.append(val_tpr)
        
        # X_test probabilities
        probs.loc[:, 'Fold_{}_Prob_0'.format(fold)] = model.predict_proba(X_test)[:, 0]
        probs.loc[:, 'Fold_{}_Prob_1'.format(fold)] = model.predict_proba(X_test)[:, 1]
            
        oob += model.oob_score_ / N
        print('Fold {} OOB Score: {}\n'.format(fold, model.oob_score_))   
        
    print('Average OOB Score: {}'.format(oob))
    
#%% Main

if __name__ == '__main__':
    # Import Data
    df = importData(inputTrain, inputTestX, inputTesty)
    
    # Determine best correlation for age used for imputation
    corrAge(df)
    
    # Impute age by age and sex
    df = imputeAge(df)
    
    # Impute Embark
    df = imputeEmbark(df)
    
    # Impute Fare
    df = imputeFare(df)
    
    # Impute Deck
    df = imputeDeck(df)
    
    # Split df into train and test
    df_train, df_test = divide_df(df)
    
    # Create Model
    model = createModel(SEED)
    
    # Evaluate Baseline Model
    #evaluateModel(model, X_train, y_train, X_test, y_test, N)
    





