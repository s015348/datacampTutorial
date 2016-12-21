# From:https://campus.datacamp.com
# Import the libraries
import csv as csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.tree as tree
import utility as u

# Get the model
train, test = u.load_data()
train = u.prepare_data(train)

# Create a new array with the added features: features_two
#print(train.describe())
feature_list = ["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "PassengerId"]

#Control overfitting by setting "max_depth" to 10 and "min_samples_split" to 5 : my_tree_two
max_depth = 5
min_samples_split = 1000
my_tree, features, target = u.train_model(train, feature_list)


#Print the score of the new decison tree
# Look at the importance and score of the included features
print(my_tree.feature_importances_)
print(my_tree.score(features, target))

plt.figure(1)
plt.xlabel("Features")
plt.ylabel("Importances")
plt.title("Importance and score of the included features")
plt.xticks(np.arange(len(feature_list)), feature_list)
plt.plot(my_tree.feature_importances_, 'ro')
score_str = "Score:" + str(my_tree.score(features, target))
plt.text(2, 0.2, score_str, fontsize=15, verticalalignment="top", horizontalalignment="left")
plt.show()