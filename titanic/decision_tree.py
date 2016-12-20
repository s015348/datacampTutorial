# From:https://campus.datacamp.com
# Import the libraries
import csv as csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.tree as tree



def print_features(data, f_x, f_y, title, index=111, color_opt='bo'):
    plt.subplot(index)
    plt.xlabel(f_x)
    plt.ylabel(f_y)
    plt.title(title)
    plt.plot(data[f_x], data[f_y], color_opt)


# Load the train and test datasets to create two DataFrames
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Convert the male and female groups to integer form
train.loc[train["Sex"] == "male", "Sex"] = 0
train.loc[train["Sex"] == "female", "Sex"] = 1

# substitute each missing value with the median of the all present values
train["Age"] = train["Age"].fillna(train["Age"].median())
train["Sex"] = train["Sex"].fillna(train["Sex"].median())

# Print the train data to see the available features
#print(train)
plt.figure(1)
print_features(train, 'Sex', 'Survived', 'Sex to surviced', 411, 'ro')
print_features(train, 'Age', 'Survived', 'Age to surviced', 412, 'go')
print_features(train, 'Fare', 'Survived', 'Fare to surviced', 413, 'bo')
print_features(train, 'Pclass', 'Survived', 'Pclass to surviced', 414, 'yo')
plt.show()

# Create the target and features numpy arrays: target, features_one
target = train["Survived"].values
features_one = train[["Sex", "Age", "Fare"]].values
#features_one = train[["Pclass", "Sex", "Age", "Fare"]].values

# Fit your first decision tree: my_tree_one
my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(features_one, target)

# Look at the importance and score of the included features
print(my_tree_one.feature_importances_)
print(my_tree_one.score(features_one, target))


