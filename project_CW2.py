# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 22:50:38 2023

Cousework 2 - Machine Learning Project

Any predictive modeling machine learning project can be broken down into six common tasks:

1. Define Problem.
2. Summarize Data.
3. Prepare Data.
4. Evaluate Algorithms.
5. Improve Results.
6. Present Results.

@author: Quyen Hong Dang - 13022961
"""

#%% 1. Prepare Problem
# a) Load libraries
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot
from pandas import set_option
from scipy.stats import f_oneway
from scipy.stats import chi2_contingency
from sklearn.preprocessing import MinMaxScaler
from numpy import set_printoptions
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from pickle import dump
from pickle import load

#%% b) Load dataset
file_1 = "CreditRiskData.csv"
load_data = open(file_1,'rt')
data = pd.read_csv(load_data)
data.dropna()
peek = data.head(20)
print(peek)

# Dimensions of data
shape = data.shape
print(shape)

# Data type for each attribute
types = data.dtypes
print(types)

#%% 2. Summarize Data
# a) Descriptive statistics
set_option('display.width', 100)
set_option('display.precision', 3)
description = data.describe(include='all')
print(description)

# Class Distribution
class_counts= data.groupby('GoodCredit').size()
print(class_counts)

# Finding unique values for each attribute
n=data.nunique()
print(n)

#%% b) Data visualizations

# Plotting histograms for continuous variables
data.hist(['duration', 'amount','age'], figsize=(12,8))
plt.show()

# Box plots for categorical "GoodCredit" and continuous predictors
ContinuousColsList=['duration','amount', 'age']

fig, PlotCanvas=plt.subplots(nrows=1, ncols=len(ContinuousColsList), figsize=(12,5))
for PredictorCol , i in zip(ContinuousColsList, range(len(ContinuousColsList))):
    data.boxplot(column=PredictorCol, by='GoodCredit', 
                 figsize=(5,5), vert=True, ax=PlotCanvas[i])
    
#%% Plotting bar charts for categorical variables
def PlotBarCharts(inpData, colsToPlot):
    fig, subPlot=plt.subplots(nrows=1, ncols=len(colsToPlot), figsize=(12,3))
    fig.suptitle('Bar charts of: '+ str(colsToPlot))

    for colName, plotNumber in zip(colsToPlot, range(len(colsToPlot))):
        inpData.groupby(colName).size().plot(kind='bar',ax=subPlot[plotNumber])

PlotBarCharts(inpData=data, 
              colsToPlot=['checkingstatus', 'history', 'purpose','savings','employ'])
PlotBarCharts(inpData=data, 
              colsToPlot=['installment', 'status', 'others','residence','property','otherplans'])
PlotBarCharts(inpData=data, 
              colsToPlot=['housing', 'cards', 'job', 'liable', 'tele', 'foreign'])


# Box plots for categorical "GoodCredit" and categorical predictors
CategoricalColsList=['checkingstatus', 'history', 'purpose','savings','employ',
                     'installment', 'status', 'others','residence', 'property',
                     'otherplans', 'housing', 'cards', 'job', 'liable', 'tele', 'foreign']

fig, PlotCanvas=plt.subplots(nrows=len(CategoricalColsList), ncols=1, figsize=(12,80))

for CategoricalCol , i in zip(CategoricalColsList, range(len(CategoricalColsList))):
    CrossTabResult=pd.crosstab(index=data[CategoricalCol], columns=data['GoodCredit'])
    CrossTabResult.plot.bar(color=['blue','pink'], ax=PlotCanvas[i])

#%% 3. Prepare Data
# a) Data Cleaning
data.drop_duplicates()
missingvalues= data.isnull().sum()
print(missingvalues)
    
#%% b) Feature Selection
#% Statistical Feature Selection (Categorical Vs Continuous) using ANOVA test
def FunctionAnova(inpData, TargetVariable, ContinuousPredictorList):
    SelectedPredictors=[]
    
    for predictor in ContinuousPredictorList:
        CategoryGroupLists=inpData.groupby(TargetVariable)[predictor].apply(list)
        AnovaResults = f_oneway(*CategoryGroupLists)
        
        # If the ANOVA P-Value is <0.05, that means we reject H0
        if (AnovaResults[1] < 0.05):
            print(predictor, 'is correlated with', TargetVariable, '| P-Value:', AnovaResults[1])
            SelectedPredictors.append(predictor)
        else:
            print(predictor, 'is NOT correlated with', TargetVariable, '| P-Value:', AnovaResults[1])
    
    return(SelectedPredictors)

ContinuousVariables=['duration', 'amount','age']
FunctionAnova(inpData=data, TargetVariable='GoodCredit', ContinuousPredictorList=ContinuousVariables)

#%%Statistical Feature Selection (Categorical Vs Categorical) using Chi-Square Test
def FunctionChisq(inpData, TargetVariable, CategoricalVariablesList):
    SelectedPredictors=[]

    for predictor in CategoricalVariablesList:
        CrossTabResult=pd.crosstab(index=inpData[TargetVariable], columns=inpData[predictor])
        ChiSqResult = chi2_contingency(CrossTabResult)
        
        # If the ChiSq P-Value is <0.05, that means we reject H0
        if (ChiSqResult[1] < 0.05):
            print(predictor, 'is correlated with', TargetVariable, '| P-Value:', ChiSqResult[1])
            SelectedPredictors.append(predictor)
        else:
            print(predictor, 'is NOT correlated with', TargetVariable, '| P-Value:', ChiSqResult[1])        
            
    return(SelectedPredictors)

CategoricalVariables=['checkingstatus', 'history', 'purpose','savings','employ',
                     'installment', 'status', 'others','residence', 'property',
                     'otherplans', 'housing', 'cards', 'job', 'liable', 'tele', 'foreign']

FunctionChisq(inpData=data, TargetVariable='GoodCredit', CategoricalVariablesList= CategoricalVariables)

#%% Selecting final predictors for Machine Learning
SelectedColumns=['checkingstatus','history','purpose','savings','employ','status','others','property','otherplans','housing','foreign','age','amount','duration']

Data=data[SelectedColumns]
peek1=Data.head(20)
print(peek1)

#%% c) Data Transforms
# Treating the Ordinal variable first
Data['employ'].replace({'A71':1, 'A72':2,'A73':3, 'A74':4,'A75':5 }, inplace=True)

# Treating the binary nominal variable
Data['foreign'].replace({'A201':1, 'A202':0}, inplace=True)

# Treating all the nominal variables at once using dummy variables
Data_Numeric=pd.get_dummies(Data)

# Adding Target Variable to the data
Data_Numeric['GoodCredit']=data['GoodCredit']

# Looking data after treatment
peek2= Data_Numeric.head()
print(peek2)

#%% Printing all the column names for our reference
peek3 = Data_Numeric.columns
print(peek3)

# Separate Target Variable and Predictor Variables
TargetVariable='GoodCredit'
Predictors=['employ', 'foreign', 'age', 'amount', 'duration', 'checkingstatus_A11',
       'checkingstatus_A12', 'checkingstatus_A13', 'checkingstatus_A14', 'savings_A61',
       'savings_A62', 'savings_A63', 'savings_A64', 'savings_A65',
       'history_A30', 'history_A31', 'history_A32', 'history_A33',
       'history_A34', 'purpose_A40', 'purpose_A41', 'purpose_A410',
       'purpose_A42', 'purpose_A43', 'purpose_A44', 'purpose_A45',
       'purpose_A46', 'purpose_A48', 'purpose_A49',
       'status_A91', 'status_A92', 'status_A93', 'status_A94', 'others_A101',
       'others_A102', 'others_A103', 'property_A121', 'property_A122',
       'property_A123', 'property_A124', 'otherplans_A141', 'otherplans_A142',
       'otherplans_A143', 'housing_A151', 'housing_A152', 'housing_A153']

X=Data_Numeric[Predictors].values
Y=Data_Numeric[TargetVariable].values

scaler=MinMaxScaler()
rescaledX=scaler.fit_transform(X)
set_printoptions(precision=3)
print(rescaledX[0:5,:])

#%% 4. Evaluate Algorithms
# a) Split-out validation dataset
X_train, X_test, y_train, y_test = train_test_split(rescaledX, Y, test_size=0.3, random_state=428)
# Sanity check for the sampled data
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

#%% b) Test options and evaluation metric
num_folds=10
seed=428
scoring='accuracy'

#%% c) Spot Check Algorithms
models=[]
models.append(('LR', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART',DecisionTreeClassifier()))
models.append(('SVM', SVC()))
models.append(('NB', GaussianNB()))

# evaluating each model in turns
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=num_folds, random_state=seed, shuffle = True) 
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg='%s: %f (%f)' % (name, cv_results.mean(), cv_results.std())
    print(msg)
#%% d) Compare Algorithms
fig=pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax=fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

#%% 5. Improve Accuracy
# a) Algorithm Tuning
c_values=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1,3, 1.5, 1.7, 2.0]
kernel_values = ['linear', 'poly', 'rbf', 'sigmoid']
param_grid =dict(C=c_values, kernel=kernel_values)
model = SVC(gamma='auto')
kfold = KFold(n_splits=num_folds, random_state=seed, shuffle = True)
grid=GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result=grid.fit(X_train, y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds= grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean,stdev,param))
    
#%% b) Ensembles
ensembles=[]
ensembles.append(('AB', AdaBoostClassifier()))
ensembles.append(('GBM', GradientBoostingClassifier()))
ensembles.append(('RF', RandomForestClassifier(n_estimators=10)))
ensembles.append(('ET', ExtraTreesClassifier(n_estimators=10)))
results1=[]
names1=[]
for name, model in ensembles:
    kfold = KFold(n_splits=num_folds, random_state=seed,shuffle=True)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results1.append(cv_results)
    names1.append(name)
    msg='%s: %f (%f)' % (name, cv_results.mean(), cv_results.std())
    print(msg)
# Compare Algorithms
fig=pyplot.figure()
fig.suptitle('Ensemble Algorithm Comparison')
ax=fig.add_subplot(111)
pyplot.boxplot(results1)
ax.set_xticklabels(names1)
pyplot.show()
#%% 6. Finalize Model
# a) Prepare the model
model = SVC(C=3)
model.fit(X_train, y_train)
# b) Make predictions on valiation dataset
predictions = model.predict(X_test)
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
#%% c) Save model for later use
# save the model to disk
filename = 'finalized_model.sav'
dump(model,open(filename,'wb'))
# load the model from disk
loaded_model = load(open(filename,'rb'))
result = loaded_model.score(X_test,y_test)
print(result)
