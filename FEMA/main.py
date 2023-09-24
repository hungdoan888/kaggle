#%% Libraries

import pandas as pd

#%% Variables

data_filepath = 'data/DisasterDeclarationsSummaries.csv'

#%% Read Data

df = pd.read_csv(data_filepath)

#%% Get some info on this dataset

# Get the shape of the dataframe
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

# Datatypes and non-null counts
print(df.info())
"""incidentEndDate, disasterCloseoutDate, lastIAFileingDate have null values"""

# Get a statistical summary of each column
print(df.describe())

# Loop through each categorical column and display its value counts
for column in df.select_dtypes(include=['object']).columns:
    print(column)
    print(df[column].value_counts())
    print("-----")

# Null Values
print(df.isnull().sum())

# Duplicates
duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")

# Outliers
for column in df.select_dtypes(include=['float64', 'int64']).columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR)))
    print(f"{column} has {outliers.sum()} potential outliers")
