# From:https://campus.datacamp.com
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.tree as tree

def load_data():
    # Load the train and test datasets to create two DataFrames
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    return train, test

def prepare_data(data):
    # substitute each missing value with the median of the all present values
    # columns are "Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"
    data["Pclass"] = data["Pclass"].fillna(3)
    data["Age"] = data["Age"].fillna(data["Age"].median())
    data["Sex"] = data["Sex"].fillna("male")
    data["Fare"] = data["Fare"].fillna(data["Fare"].median())
    data["SibSp"] = data["SibSp"].fillna(0)
    data["Parch"] = data["Parch"].fillna(0)
    data["Embarked"] = data["Embarked"].fillna("S")
    
    # Convert the male and female groups to integer form
    data.loc[data["Sex"] == "male", "Sex"] = 0
    data.loc[data["Sex"] == "female", "Sex"] = 1
    
    # Convert the Embarked classes to integer form
    data.loc[data["Embarked"] == "S", "Embarked"] = 0
    data.loc[data["Embarked"] == "C", "Embarked"] = 1
    data.loc[data["Embarked"] == "Q", "Embarked"] = 2 
    return data


def predict_result(test, feature_list, my_tree):
    # Extract the features from the test set: Pclass, Sex, Age, and Fare.
    test_features = test[feature_list].values
# Make your prediction using the test set
    my_prediction = my_tree.predict(test_features)
    return my_prediction


def plot_result(feature_list, my_tree, features, target, fig_num=1):
    plt.figure(fig_num)
    plt.xlabel("Features")
    plt.ylabel("Importances")
    plt.title("Importance and score of the included features")
    plt.xticks(np.arange(len(feature_list)), feature_list)
    plt.plot(my_tree.feature_importances_, 'ro')
    score_str = "Score:" + str(my_tree.score(features, target))
    plt.text(1, 0.2, score_str, fontsize=15, verticalalignment="top", horizontalalignment="left")
    plt.show()
    
        
def train_model(train, feature_list, max_depth = None, min_samples_split = 2):
    # Create the target and features numpy arrays: target, features
    target = train["Survived"].values
    features = train[feature_list].values
    
    # Fit your first decision tree: my_tree
    my_tree = tree.DecisionTreeClassifier(max_depth = max_depth, min_samples_split = min_samples_split, random_state = 1)
    my_tree = my_tree.fit(features, target)
    return my_tree, features, target
