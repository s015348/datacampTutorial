# From:https://campus.datacamp.com
# Import the Pandas library
import csv as csv
import numpy as np
import pandas as pd
from sklearn import tree

# Load the train and test datasets to create two DataFrames
train = pd.read_csv('../titanic/train.csv')
test = pd.read_csv('../titanic/test.csv')


# Create a copy of test: test_one
test_one = test

# Initialize a Survived column to 0
test_one["Survived"] = 0

# Set Survived to 1 if Sex equals "female" and print the `Survived` column from `test_one`
test_one.loc[test_one["Sex"] == "female", "Survived"] = 1
#print(test_one.describe())

# substitute each missing value with the median of the all present values
train["Age"] = train["Age"].fillna(train["Age"].median())
#print(train["Age"].median())
#print(train["Age"].describe())

# Convert the male and female groups to integer form
train.loc[train["Sex"] == "male", "Sex"] = 0
train.loc[train["Sex"] == "female", "Sex"] = 1
#print(train["Sex"])

# Impute the Embarked variable
train["Embarked"] = train["Embarked"].fillna('S')

# Convert the Embarked classes to integer form
train.loc[train["Embarked"] == "S", "Embarked"] = 0
train.loc[train["Embarked"] == "C", "Embarked"] = 1
train.loc[train["Embarked"] == "Q", "Embarked"] = 2 

#Print the Sex and Embarked columns
print(train["Embarked"])