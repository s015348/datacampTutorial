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
#plt.show()

# Create the target and features numpy arrays: target, features
target = train["Survived"].values
features = train[["Pclass", "Sex", "Age", "Fare"]].values

# Fit your first decision tree: my_tree
my_tree = tree.DecisionTreeClassifier()
my_tree = my_tree.fit(features, target)

# Look at the importance and score of the included features
print(my_tree.feature_importances_)
print(my_tree.score(features, target))

plt.figure(2)
plt.xlabel("Features")
plt.ylabel("Importances")
plt.title("Importance and score of the included features")
plt.xticks(np.arange(4), ("Pclass", "Sex", "Age", "Fare") )
plt.plot(my_tree.feature_importances_, 'ro')
score_str = "Score:" + str(my_tree.score(features, target))
plt.text(1, 0.2, score_str, fontsize=15, verticalalignment="top", horizontalalignment="left")
plt.show()
