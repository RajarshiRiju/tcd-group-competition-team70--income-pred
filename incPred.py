#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
#from sklearn.linear_model import LinearRegression


# Read the CSVs
df = pd.read_csv("tcd-ml-1920-group-income-train.csv")
df_test = pd.read_csv("tcd-ml-1920-group-income-test.csv")

df.head()

# Pre-process the train and test dataset
df = df.drop("Instance",axis = 1)
df_test = df_test.drop("Instance",axis = 1)

df = df.replace("#NUM!", np.nan)
df_test = df_test.replace("#NUM!", np.nan)

# Function to calculate target encoding of a specific column
def calc_target_encoding(df,bycolumn,ontarget, w):
    # Calculate the overall mean
    mean = df[ontarget].mean()
    print("----------------------------------")
    # Calculate the number of values and the mean of each group
    agg = df.groupby(bycolumn)[ontarget].agg(['count', 'mean'])
    counts = agg['count']
    means = agg['mean']

    # Compute the smoothed means
    smooth = (counts * means + w * mean) / (counts + w)

    # Replace each value by its smoothed mean
    return df[bycolumn].map(smooth)


df.isnull().sum()

# Renaming few columns for ease of use
df = df.rename(index=str, columns={"Year of Record": "YearofRecord"})
df_test = df_test.rename(index=str, columns={"Year of Record": "YearofRecord"})

df = df.rename(index=str, columns={"Housing Situation": "HousingSituation"})
df_test = df_test.rename(index=str, columns={"Housing Situation": "HousingSituation"})

df = df.rename(index=str, columns={"Crime Level in the City of Employement": "CrimeLevelintheCityofEmployement"})
df_test = df_test.rename(index=str, columns={"Crime Level in the City of Employement": "CrimeLevelintheCityofEmployement"})

df = df.rename(index=str, columns={"Work Experience in Current Job [years]": "WorkExperience"})
df_test = df_test.rename(index=str, columns={"Work Experience in Current Job [years]": "WorkExperience"})

df = df.rename(index=str, columns={"Satisfation with employer": "Satisfationwithemployer"})
df_test = df_test.rename(index=str, columns={"Satisfation with employer": "Satisfationwithemployer"})

df = df.rename(index=str, columns={"Size of City": "SizeofCity"})
df_test = df_test.rename(index=str, columns={"Size of City": "SizeofCity"})

df = df.rename(index=str, columns={"University Degree": "UniversityDegree"})
df_test = df_test.rename(index=str, columns={"University Degree": "UniversityDegree"})

df = df.rename(index=str, columns={"Wears Glasses": "WearsGlasses"})
df_test = df_test.rename(index=str, columns={"Wears Glasses": "WearsGlasses"})

df = df.rename(index=str, columns={"Hair Color": "HairColor"})
df_test = df_test.rename(index=str, columns={"Hair Color": "HairColor"})

df = df.rename(index=str, columns={"Body Height [cm]": "BodyHeight"})
df_test = df_test.rename(index=str, columns={"Body Height [cm]": "BodyHeight"})

df = df.rename(index=str, columns={"Yearly Income in addition to Salary (e.g. Rental Income)": "YearlyIncomeinadditiontoSalary"})
df_test = df_test.rename(index=str, columns={"Yearly Income in addition to Salary (e.g. Rental Income)": "YearlyIncomeinadditiontoSalary"})

df = df.rename(index=str, columns={"Total Yearly Income [EUR]": "TotalYearlyIncome"})
df_test = df_test.rename(index=str, columns={"Total Yearly Income [EUR]": "TotalYearlyIncome"})

# Concat Train and Test datasets
data = pd.concat([df, df_test], sort=False)

# Process each column
data['YearofRecord'] = data['YearofRecord'].fillna((data['YearofRecord'].mode()[0]))

data['YearlyIncomeinadditiontoSalary'] = data['YearlyIncomeinadditiontoSalary'].map(lambda x: x.rstrip('EUR'))
data['YearlyIncomeinadditiontoSalary'] = data['YearlyIncomeinadditiontoSalary'].astype(float)

# Target encoding columns
data['HousingSituation'] = calc_target_encoding(data,'HousingSituation','TotalYearlyIncome',48)

data['Satisfationwithemployer'] = calc_target_encoding(data,'Satisfationwithemployer','TotalYearlyIncome',48)
data['Satisfationwithemployer'] = data['Satisfationwithemployer'].fillna((data['Satisfationwithemployer'].mean()))

data['Gender'] = calc_target_encoding(data,'Gender','TotalYearlyIncome',48)
data['Gender'] = data['Gender'].fillna((data['Gender'].mean()))

data['Country'] = data['Country'].fillna((data['Country'].mode()[0]))
data['Country'] = calc_target_encoding(data,'Country','TotalYearlyIncome',48)
#data['Country'] = data['Country'].fillna((data['Country'].mean()))

data = data.drop("HairColor",axis = 1)

data['Profession'] = calc_target_encoding(data,'Profession','TotalYearlyIncome',48)
data['Profession'] = data['Profession'].fillna((data['Profession'].mean()))

data['UniversityDegree'] = calc_target_encoding(data,'UniversityDegree','TotalYearlyIncome',48)
data['UniversityDegree'] = data['UniversityDegree'].fillna((data['UniversityDegree'].mean()))
data.isnull().sum()

data['WorkExperience'] = calc_target_encoding(data,'WorkExperience','TotalYearlyIncome',48)
data['WorkExperience'] = data['WorkExperience'].fillna((data['WorkExperience'].mean()))

data.isnull().sum()

# Split back train and test data
df_new = data[0:len(df)]
df_test_new = data[len(df):]
df_new = pd.DataFrame(df_new)
df_test_new = pd.DataFrame(df_test_new)

X = df_new.drop("TotalYearlyIncome",axis=1)
Y = df_new.TotalYearlyIncome

# Split train and holdout data (80-20)
X_train,X_holdOut,Y_train,Y_holdOut = train_test_split(X,Y,test_size = 0.2, random_state=100)

# Train model
cb_model = CatBoostRegressor(iterations=10000,
                             learning_rate=0.01,
                             depth=12,
                             eval_metric='MAE',
                             random_seed = 25,
                             bagging_temperature = 0.2,
                             od_type='Iter',
                             metric_period = 75,
                             od_wait=100)
cb_model.fit(X_train, Y_train,
             eval_set=(X_holdOut,Y_holdOut),
             use_best_model=True,
             verbose=True)

# Predict on holdout data
Y_predicted = cb_model.predict(X_holdOut)

# calculate MAE
from sklearn.metrics import mean_absolute_error
meanabsoluteerror = mean_absolute_error(Y_test,Y_predicted)
print(meanabsoluteerror)

# Final predictions
X_question_test = df_test_new.drop("TotalYearlyIncome", axis=1)
cb_model.fit(X,Y)
Y_question_pred = cb_model.predict(X_question_test)
Y_question_pred = pd.DataFrame(Y_question_pred)
Y_question_pred.to_csv("Submission.csv", sep=',', index=False, header=True)

