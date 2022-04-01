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

inputTrain = r'.\input\train.csv'
inputTestX = r'.\input\test.csv'
inputTesty = r'.\input\gender_submission.csv'

# Define global variables
SEED = 42
N = 5

#%% Suppress Warnings

warnings.filterwarnings('ignore')

#%% Seaborn Settings

sns.set(style="darkgrid")

#%% Divide df

def concat_df(train_data, test_data):
    # Returns a concatenated df of training and test set
    return pd.concat([train_data, test_data], sort=True).reset_index(drop=True)

def divide_df(df):
    # Returns divided dfs of training and test set
    return df.loc[:890], df.loc[891:]

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

#%% Bin Fare

def binFare(df):
    df['Fare'] = pd.qcut(df['Fare'], 13)
    return df

#%% Bin Age

def binAge(df):
    df['Age'] = pd.qcut(df['Age'], 10)
    return df

#%% Bin Family Size

def binFamilySize(df):
    # Siblings + Spouse + Parents + self
    df['Family_Size'] = df['SibSp'] + df['Parch'] + 1
    
    # Map Family size to categorical
    family_map = {1: 'Alone', 2: 'Small', 3: 'Small', 4: 'Small', 5: 'Medium', 6: 'Medium', 7: 'Large', 8: 'Large', 
                  11: 'Large'}
    df['Family_Size_Grouped'] = df['Family_Size'].map(family_map)
    return df

#%% Bin Ticket Frequency

def binTicketFrequency(df):
    df['Ticket_Frequency'] = df.groupby('Ticket')['Ticket'].transform('count')
    return df

#%% Bin Titles

def binTitles(df):
    df['Title'] = df['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]
    df['Is_Married'] = 0
    df['Is_Married'].loc[df['Title'] == 'Mrs'] = 1
    
    df['Title'] = df['Title'].replace(['Miss', 'Mrs','Ms', 'Mlle', 'Lady', 'Mme', 'the Countess', 'Dona'], 
                                      'Miss/Mrs/Ms')
    df['Title'] = df['Title'].replace(['Dr', 'Col', 'Major', 'Jonkheer', 'Capt', 'Sir', 'Don', 'Rev'], 
                                      'Dr/Military/Noble/Clergy')
    return df
 
#%% Bin Family Survival Rate

def binFamilySurvivalRate(df):
    # Create family surname from name
    df['Family'] = extract_surname(df['Name'])
    
    # Split df into train and test
    df_train, df_test = divide_df(df)
    
    # Calculate Family Survival Rate in training data
    family_rates = calcFamSurvivalRate(df_train, df_test)
    
    # Add Family Survival Rates to training and test data
    df_train, df_test = addFamSurvivalRates(df_train, df_test, family_rates)
    return df_train, df_test

# Extract Surname
def extract_surname(data):       
    families = []   
    for i in range(len(data)):        
        name = data.iloc[i]
        if '(' in name:
            name_no_bracket = name.split('(')[0] 
        else:
            name_no_bracket = name            
        family = name_no_bracket.split(',')[0]       
        for c in string.punctuation:
            family = family.replace(c, '').strip()           
        families.append(family)           
    return families

# Calculate Family Survival Rate in training data
def calcFamSurvivalRate(df_train, df_test):
    # Creating a list of families and tickets that are occuring in both training and test set
    non_unique_families = [x for x in df_train['Family'].unique() if x in df_test['Family'].unique()]
    df_family_survival_rate = df_train.groupby('Family')['Survived', 'Family','Family_Size'].median()

#%% Evaulaute Model
    family_rates = {}
    for i in range(len(df_family_survival_rate)):
        # Checking a family exists in both training and test set, and has members more than 1
        if df_family_survival_rate.index[i] in non_unique_families and df_family_survival_rate.iloc[i, 1] > 1:
            family_rates[df_family_survival_rate.index[i]] = df_family_survival_rate.iloc[i, 0]
    return family_rates

# Add Family Survival Rates to training and test data
def addFamSurvivalRates(df_train, df_test, family_rates):
    mean_survival_rate = np.mean(df_train['Survived'])

    train_family_survival_rate = []
    train_family_survival_rate_NA = []
    test_family_survival_rate = []
    test_family_survival_rate_NA = []
    
    for i in range(len(df_train)):
        if df_train['Family'][i] in family_rates:
            train_family_survival_rate.append(family_rates[df_train['Family'][i]])
            train_family_survival_rate_NA.append(1)
        else:
            train_family_survival_rate.append(mean_survival_rate)
            train_family_survival_rate_NA.append(0)
            
    for i in range(len(df_test)):
        if df_test['Family'].iloc[i] in family_rates:
            test_family_survival_rate.append(family_rates[df_test['Family'].iloc[i]])
            test_family_survival_rate_NA.append(1)
        else:
            test_family_survival_rate.append(mean_survival_rate)
            test_family_survival_rate_NA.append(0)
            
    df_train['Family_Survival_Rate'] = train_family_survival_rate
    df_train['Family_Survival_Rate_NA'] = train_family_survival_rate_NA
    df_test['Family_Survival_Rate'] = test_family_survival_rate
    df_test['Family_Survival_Rate_NA'] = test_family_survival_rate_NA
    return df_train, df_test

#%% Bin ticket Survival Rate

def binTicketSurvivalRate(df_train, df_test):
    # Calculate Ticket Survival Rate in training data
    ticket_rates = calcTicketSurvivalRates(df_train, df_test)
    
    # Add Ticket Survival Rates to training and test data
    df_train, df_test = addTicketSurvivalRates(df_train, df_test, ticket_rates)
    return df_train, df_test

# Calculate Ticket Survival Rate in training data
def calcTicketSurvivalRates(df_train, df_test):
    non_unique_tickets = [x for x in df_train['Ticket'].unique() if x in df_test['Ticket'].unique()]
    df_ticket_survival_rate = df_train.groupby('Ticket')['Survived', 'Ticket','Ticket_Frequency'].median()
    
    ticket_rates = {}
    for i in range(len(df_ticket_survival_rate)):
        # Checking a ticket exists in both training and test set, and has members more than 1
        if df_ticket_survival_rate.index[i] in non_unique_tickets and df_ticket_survival_rate.iloc[i, 1] > 1:
            ticket_rates[df_ticket_survival_rate.index[i]] = df_ticket_survival_rate.iloc[i, 0]
    return ticket_rates

# Add Ticket Survival Rates to training and test data
def addTicketSurvivalRates(df_train, df_test, ticket_rates):
    mean_survival_rate = np.mean(df_train['Survived'])
    
    train_ticket_survival_rate = []
    train_ticket_survival_rate_NA = []
    test_ticket_survival_rate = []
    test_ticket_survival_rate_NA = []
    
    for i in range(len(df_train)):
        if df_train['Ticket'][i] in ticket_rates:
            train_ticket_survival_rate.append(ticket_rates[df_train['Ticket'][i]])
            train_ticket_survival_rate_NA.append(1)
        else:
            train_ticket_survival_rate.append(mean_survival_rate)
            train_ticket_survival_rate_NA.append(0)
            
    for i in range(len(df_test)):
        if df_test['Ticket'].iloc[i] in ticket_rates:
            test_ticket_survival_rate.append(ticket_rates[df_test['Ticket'].iloc[i]])
            test_ticket_survival_rate_NA.append(1)
        else:
            test_ticket_survival_rate.append(mean_survival_rate)
            test_ticket_survival_rate_NA.append(0)
            
    df_train['Ticket_Survival_Rate'] = train_ticket_survival_rate
    df_train['Ticket_Survival_Rate_NA'] = train_ticket_survival_rate_NA
    df_test['Ticket_Survival_Rate'] = test_ticket_survival_rate
    df_test['Ticket_Survival_Rate_NA'] = test_ticket_survival_rate_NA
    return df_train, df_test

#%% Calculate overall survival rate

def calcOverallSurvival(df_train, df_test):
    for df in [df_train, df_test]:
        df['Survival_Rate'] = (df['Ticket_Survival_Rate'] + df['Family_Survival_Rate']) / 2
        df['Survival_Rate_NA'] = (df['Ticket_Survival_Rate_NA'] + df['Family_Survival_Rate_NA']) / 2
    return df_train, df_test

#%% Label Encoding

def labelEncoding(df_train, df_test):
    non_numeric_features = ['Embarked', 'Sex', 'Deck', 'Title', 'Family_Size_Grouped', 'Age', 'Fare']

    for df in [df_train, df_test]:
        for feature in non_numeric_features:        
            df[feature] = LabelEncoder().fit_transform(df[feature])
    return df_train, df_test

#%% One hot Encoding

def oneHotEncoding(df_train, df_test):
    cat_features = ['Pclass', 'Sex', 'Deck', 'Embarked', 'Title', 'Family_Size_Grouped']
    encoded_features = []
    
    for df in [df_train, df_test]:
        for feature in cat_features:
            encoded_feat = OneHotEncoder().fit_transform(df[feature].values.reshape(-1, 1)).toarray()
            n = df[feature].nunique()
            cols = ['{}_{}'.format(feature, n) for n in range(1, n + 1)]
            encoded_df = pd.DataFrame(encoded_feat, columns=cols)
            encoded_df.index = df.index
            encoded_features.append(encoded_df)
    
    df_train = pd.concat([df_train, *encoded_features[:6]], axis=1)
    df_test = pd.concat([df_test, *encoded_features[6:]], axis=1)
    return df_train, df_test

#%% Drop useless columns

def dropUselessColumns(df_train, df_test):
    keep_cols = ['Age',	'Fare', 'Is_Married', 'Ticket_Frequency',
                 'Deck_1', 'Deck_2', 'Deck_3', 'Deck_4',
                 'Embarked_1', 'Embarked_2','Embarked_3',
                 'Family_Size_Grouped_1', 'Family_Size_Grouped_2', 'Family_Size_Grouped_3', 'Family_Size_Grouped_4', 
                 'Pclass_1', 'Pclass_2', 'Pclass_3',
                 'Sex_1', 'Sex_2',
                 'Survival_Rate', 'Survival_Rate_NA',  
                 'Title_1',	'Title_2', 'Title_3', 'Title_4',
                 'Survived']
    df_train = df_train[keep_cols]
    df_test = df_test[keep_cols]
    return df_train, df_test

#%% Use a standard scalar to scale variables

def useStandardScalar(df_train, df_test):
    # Train
    X_train = StandardScaler().fit_transform(df_train.drop(columns='Survived'))
    y_train = df_train['Survived'].values
    
    # Test
    X_test = StandardScaler().fit_transform(df_test.drop(columns='Survived'))
    y_test = df_test['Survived'].values
    return X_train, y_train, X_test, y_test

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
    columns.remove('Survived')
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
    # Class Survived are columns that end with Prob_1
    class_survived = [col for col in df_pred.columns if col.endswith('Prob_1')]
    
    # Average Survived and not survived classes
    df_pred['1'] = df_pred[class_survived].sum(axis=1) / N
    df_pred['0'] = df_pred.drop(columns=class_survived).sum(axis=1) / N
    
    # Initiate prediction column
    df_pred['pred'] = 0
    
    # Get indices of survived and set pred = 1 if survived
    pos = df_pred[df_pred['1'] >= 0.5].index
    df_pred.loc[pos, 'pred'] = 1
    
    # Predictions
    y_pred = df_pred['pred'].astype(int)
    
    # Get test df
    _, df_test = divide_df(df)
    
    # Create submissions df with PassengerId and Survived
    df_submission = pd.DataFrame()
    df_submission['PassengerId'] = df_test['PassengerId']
    df_submission['Predicted'] = y_pred.values
    df_submission['Actual'] = y_test
    
    # Count how many correct
    df_submission['Correct'] = df_submission.apply(lambda row: 1 if row['Predicted'] == row['Actual'] else 0, axis=1)
    
    acc = df_submission['Correct'].sum() / df_submission['Correct'].count()
    return df_submission, acc

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
    
    # Bin Fare
    df = binFare(df)
    
    # Bin Age
    df = binAge(df)
    
    # Bin Family Size
    df = binFamilySize(df)
    
    # Bin Ticket Frequency
    df = binTicketFrequency(df)
    
    # Bin Titles
    df = binTitles(df) 
    
    # Bin Family Survival Rate
    df_train, df_test = binFamilySurvivalRate(df)
    
    # Bin Ticket Survival Rate
    df_train, df_test = binTicketSurvivalRate(df_train, df_test)

    # Calculate overall survival rate
    df_train, df_test = calcOverallSurvival(df_train, df_test)
    
    # Label Encoding for non-numeric features
    df_train, df_test = labelEncoding(df_train, df_test)
    
    # One hot encode categorical features
    df_train, df_test = oneHotEncoding(df_train, df_test)
    
    # Drop useless columns
    df_train, df_test = dropUselessColumns(df_train, df_test)
    
    # Use standard scalar
    X_train, y_train, X_test, y_test = useStandardScalar(df_train, df_test)
    
    # Create Model
    model = createModel(SEED)
    
    # Evaluate Model
    df_pred, df_roc, df_featImp, oob = evaluateModel(model, df_train, X_train, y_train, X_test, N)
    
    # Plot ROC Curve
    plot_roc_curve(df_roc)
    
    # Plot Feature Importance
    plotFeatImp(df_featImp)
    
    # Make Prediction
    df_submission, acc = makePrediction(model, df_pred, df, y_test, N)





