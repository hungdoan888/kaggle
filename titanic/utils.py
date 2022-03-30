# -*- coding: utf-8 -*-

#%% Libraries

# Pandas/Numpy
import numpy as np
import pandas as pd

# Imputation 
from sklearn.linear_model import LinearRegression
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

# Outlier Detection
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

# Feature Selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif

# Dimensionality Reduction
from numpy import mean
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
        
#%% Impute values

class Imputation:
    # KNN Imputer
    def imputeValuesKNN(self, X):
        # Impute values
        imputer = KNNImputer(n_neighbors=5)
        xKnn = imputer.fit_transform(X)
        
        # Create df
        X_knn = pd.DataFrame(xKnn)
        X_knn.columns = X.columns
        return X_knn
    
    # Iterative Imputer
    def imputeValuesMICE(self, X):
        # Impute values
        lr = LinearRegression()
        imp = IterativeImputer(estimator=lr,
                               missing_values=np.nan, 
                               max_iter=10, 
                               verbose=2, 
                               imputation_order='roman',
                               random_state=0)
        xMice = imp.fit_transform(X)
        
        # Create df
        X_mice = pd.DataFrame(xMice)
        X_mice.columns = X.columns
        return X_mice
    
    # Evaluate Model
    def fitModel(self, model, X, y, k, scoring='neg_mean_absolute_error'):
        # Fit model
        model.fit(X, y)
        
        # evaluate model
        n_scores = cross_val_score(model, X, y, cv=k, scoring=scoring)
        
        # Calculate mean and std
        avgResults = np.mean(abs(n_scores))   
        stdResults = np.std(abs(n_scores))
        return avgResults, stdResults
    
    # Evaluate Imputers
    def evaluateImputers(self, model, X, y, k):
        # KNN Imputer
        X_knn = self.imputeValuesKNN(X)
        avgResultsKnn, stdResultsKnn = self.fitModel(model, X_knn, y, k)
        
        # Iterative Imputer
        X_mice = self.imputeValuesMICE(X)
        avgResultsMice, stdResultsMice = self.fitModel(model, X_mice, y, k)
        
        print('\nImputers')
        print('KNN:', avgResultsKnn)
        print('MICE:', avgResultsMice)
        
#%% Outlier Detection
 
class OutlierDetection:
    # Isolation Forest      
    def isoForest(self, X, y, contamination=0.1):       
        # identify outliers in the training dataset
        iso = IsolationForest(contamination=contamination)
        yhat = iso.fit_predict(X)
        
        # select all rows that are not outliers
        X_outliersRemoved, y_outliersRemoved = self.filterByYhat(X, y, yhat)
        return X_outliersRemoved, y_outliersRemoved
    
    # Minimum Covariance Determinant
    def minCovDet(self, X, y, contamination=0.1):       
        # identify outliers in the training dataset
        ee = EllipticEnvelope(contamination=contamination)
        yhat = ee.fit_predict(X)
        
        # select all rows that are not outliers
        X_outliersRemoved, y_outliersRemoved = self.filterByYhat(X, y, yhat)
        return X_outliersRemoved, y_outliersRemoved
    
    # Local Outlier Factor
    def localOutlierDet(self, X, y):       
        # identify outliers in the training dataset
        lof = LocalOutlierFactor()
        yhat = lof.fit_predict(X)   
        
        # select all rows that are not outliers
        X_outliersRemoved, y_outliersRemoved = self.filterByYhat(X, y, yhat)
        return X_outliersRemoved, y_outliersRemoved
    
    # One class SVM
    def oneClassSVM(self, X, y, nu=0.01):       
        # identify outliers in the training dataset
        ocs = OneClassSVM(nu=nu)
        yhat = ocs.fit_predict(X)      
        
        # select all rows that are not outliers
        X_outliersRemoved, y_outliersRemoved = self.filterByYhat(X, y, yhat)
        return X_outliersRemoved, y_outliersRemoved
    
    # Filter by yhat
    def filterByYhat(self, X, y, yhat):
        mask = yhat != -1
        X_outliersRemoved = X[mask]
        y_outliersRemoved = y[mask]
        return X_outliersRemoved, y_outliersRemoved
    
    # Evaluate Model
    def fitModel(self, model, X, y, k, scoring='neg_mean_absolute_error'):
        # Fit model
        model.fit(X, y)
        
        # evaluate model
        n_scores = cross_val_score(model, X, y, cv=k, scoring=scoring)
        
        # Calculate mean and std
        avgResults = np.mean(abs(n_scores))   
        stdResults = np.std(abs(n_scores))
        return avgResults, stdResults
    
    # Evaluate all outlier detectors
    def evaluateOutlierDetectors(self, model, X, y, k, scoring='neg_mean_absolute_error'):
        # Isolation Forest
        X_isoForest, y_isoForest = self.isoForest(X, y)
        avgResultsIsoForest, stdResultsIsoForest = self.fitModel(model, X_isoForest, y_isoForest, k, scoring)
        
        # Minimum Covariance Determinant
        X_minCovDet, y_minCovDet = self.minCovDet(X, y)
        avgResultsMinCovDet, stdResultsMinCovDet = self.fitModel(model, X_minCovDet, y_minCovDet, k, scoring)
        
        # Local Outlier Factor
        X_localOutlierDet, y_localOutlierDet = self.localOutlierDet(X, y)
        avgResultsLocalOutlierDet, stdResultsLocalOutlierDet = self.fitModel(model, X_localOutlierDet, 
                                                                             y_localOutlierDet, k, scoring)
        
        # One class SVM
        X_oneClassSVM, y_oneClassSVM = self.oneClassSVM(X, y)
        avgResultsOneClassSVM, stdResultsOneClassSVM = self.fitModel(model, X_oneClassSVM, y_oneClassSVM, k, scoring)
        
        print('\nOutlier Detection')
        print('Isolated Forest:', avgResultsIsoForest)
        print('Minimum Covariance Determinant:', avgResultsMinCovDet)
        print('Local Outlier Factor:', avgResultsLocalOutlierDet)
        print('One class SVM:', avgResultsOneClassSVM)
  
#%% Feature Selection

class FeatureSelection:
    # ----------------------------------
    # Univariate Based Feature Selection
    # ----------------------------------
    
    # f_regression
    def univariateFRegression(X, y, numColsToKeep=2):
        selector = SelectKBest(f_regression, k=numColsToKeep)
        selector.fit_transform(X, y)
        
        # Create new df
        cols = selector.get_support(indices=True)
        X_univariate = X.iloc[:, cols]
        return X_univariate
    
    # mutual_info_regression
    def univariateMutInfoReg(X, y, numColsToKeep=2):
        selector = SelectKBest(mutual_info_regression, k=numColsToKeep)
        selector.fit_transform(X, y)
        
        # Create new df
        cols = selector.get_support(indices=True)
        X_univariate = X.iloc[:, cols]
        return X_univariate
    
    # chi2
    def univariateChi2(X, y, numColsToKeep=2):
        selector = SelectKBest(chi2, k=numColsToKeep)
        selector.fit_transform(X, y)
        
        # Create new df
        cols = selector.get_support(indices=True)
        X_univariate = X.iloc[:, cols]
        return X_univariate
    
    # f_classif
    def univariateFClassif(X, y, numColsToKeep=2):
        selector = SelectKBest(f_classif, k=numColsToKeep)
        selector.fit_transform(X, y)
        
        # Create new df
        cols = selector.get_support(indices=True)
        X_univariate = X.iloc[:, cols]
        return X_univariate
    
    # mutual_info_classif
    def univariateMutInfoClassif(X, y, numColsToKeep=2):
        selector = SelectKBest(mutual_info_classif, k=numColsToKeep)
        selector.fit_transform(X, y)
        
        # Create new df
        cols = selector.get_support(indices=True)
        X_univariate = X.iloc[:, cols]
        return X_univariate
    
    # -----------------
    # Select From Model
    # -----------------
    
    # L1-based feature selection
    def fromModelL1(X, y, model):
        pass
    
#%% Dimensionality Reduction

class DimensionalityReduction:
    # Principal Component Analysis (PCA)
    def drPCA(self, model, n_components=10):
        # define the pipeline
        steps = [('pca', PCA(n_components=n_components)), ('m', model)]
        model = Pipeline(steps=steps)
        return model
    
    # Singular Value Decomposition (SVD)
    def drSVD(self, model, n_components=10):
        # define the pipeline
        steps = [('svd', TruncatedSVD(n_components=n_components)), ('m', model)]
        model = Pipeline(steps=steps)
        return model
    
    # Linear Discriminant Analysis (LDA)
    def drLDA(self, model, n_components=10):
        # define the pipeline
        steps = [('lda', LinearDiscriminantAnalysis(n_components=n_components)), ('m', model)]
        model = Pipeline(steps=steps)
        return model
    
    # Isomap Embedding (ISO)
    def drISO(self, model, n_components=10):
        # define the pipeline
        steps = [('iso', Isomap(n_components=n_components)), ('m', model)]
        model = Pipeline(steps=steps)
        return model
    
    # Locally Linear Embedding (LLE)
    def drLLE(self, model, n_components=10):
        # define the pipeline
        steps = [('lle', LocallyLinearEmbedding(n_components=n_components)), ('m', model)]
        model = Pipeline(steps=steps)
        return model
    
    # Modified Locally Linear Embedding (MLLE)
    def drMLLE(self, model, n_components=10):
        # define the pipeline
        steps = [('lle', LocallyLinearEmbedding(n_components=n_components, method='modified', n_neighbors=10)), 
                 ('m', model)]
        model = Pipeline(steps=steps)
        return model
        
    # Evaluate Model
    def evaluateModel(self, model, X, y, scoring='accuracy', cv=10):
        # evaluate model
        n_scores = cross_val_score(model, X, y, scoring=scoring, cv=cv)
        return mean(abs(n_scores))
        
    # Evaluate All Models
    def evaluateAllModels(self, model, X, y, scoring='accuracy', n_components=10):
        meanPCA = self.evaluateModel(self.drPCA(model), X, y, scoring)  # PCA
        meanSVD = self.evaluateModel(self.drSVD(model), X, y, scoring)  # SVD
        meanLDA = self.evaluateModel(self.drLDA(model), X, y, scoring)  # LDA
        meanISO = self.evaluateModel(self.drISO(model), X, y, scoring)  # ISO
        meanLLE = self.evaluateModel(self.drLLE(model), X, y, scoring)  # LLE
        meanMLLE = self.evaluateModel(self.drMLLE(model), X, y, scoring)  # MLLE
        
        # Create a df of results
        df_drMean = pd.DataFrame({'PCA': [meanPCA],
                                  'SVD': [meanSVD],
                                  'LDA': [meanLDA],
                                  'ISO': [meanISO],
                                  'LLE': [meanLLE],
                                  'MLLE': [meanMLLE]})
        return df_drMean
        
