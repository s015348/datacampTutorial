# From:https://campus.datacamp.com
# Import the libraries
import csv as csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.tree as tree
import utility as u

 
def print_features(data, f_x, f_y, title, index=111, color_opt='bo'):
    plt.subplot(index)
    plt.xlabel(f_x)
    plt.ylabel(f_y)
    plt.title(title)
    plt.plot(data[f_x], data[f_y], color_opt)


train, test = u.load_data()
train = u.prepare_data(train)

# Print the train data to see the available features
#print(train)
plt.figure(1)
print_features(train, 'Sex', 'Survived', 'Sex to surviced', 411, 'ro')
print_features(train, 'Age', 'Survived', 'Age to surviced', 412, 'go')
print_features(train, 'Fare', 'Survived', 'Fare to surviced', 413, 'bo')
print_features(train, 'Pclass', 'Survived', 'Pclass to surviced', 414, 'yo')
#plt.show()

feature_list = ["Pclass", "Sex", "Age", "Fare"]
my_tree, features, target = u.train_model(train, feature_list)

# Look at the importance and score of the included features
print(my_tree.feature_importances_)
print(my_tree.score(features, target))

plt.figure(2)
plt.xlabel("Features")
plt.ylabel("Importances")
plt.title("Importance and score of the included features")
plt.xticks(np.arange(len(feature_list)), feature_list)
plt.plot(my_tree.feature_importances_, 'ro')
score_str = "Score:" + str(my_tree.score(features, target))
plt.text(1, 0.2, score_str, fontsize=15, verticalalignment="top", horizontalalignment="left")
plt.show()
