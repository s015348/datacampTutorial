# From:https://campus.datacamp.com
import pandas as pd
import sklearn.tree as tree

def load_data():
    # Load the train and test datasets to create two DataFrames
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    return train, test

def prepare_data(train):
    # Convert the male and female groups to integer form
    train.loc[train["Sex"] == "male", "Sex"] = 0
    train.loc[train["Sex"] == "female", "Sex"] = 1
    
    # substitute each missing value with the median of the all present values
    train["Age"] = train["Age"].fillna(train["Age"].median())
    train["Sex"] = train["Sex"].fillna(train["Sex"].median())
    return train


def predict_result(test, feature_list, my_tree):
    # Extract the features from the test set: Pclass, Sex, Age, and Fare.
    test_features = test[feature_list].values
# Make your prediction using the test set
    my_prediction = my_tree.predict(test_features)
    return my_prediction

    
def train_model(train, feature_list, max_depth = None, min_samples_split = 2):
    # Create the target and features numpy arrays: target, features
    target = train["Survived"].values
    features = train[feature_list].values
    
    # Fit your first decision tree: my_tree
    my_tree = tree.DecisionTreeClassifier(max_depth = max_depth, min_samples_split = min_samples_split, random_state = 1)
    my_tree = my_tree.fit(features, target)
    return my_tree, features, target
