# From:https://campus.datacamp.com
# Import the libraries
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
#import sklearn.ensemble.RandomForestClassifier as RandomForestClassifier
import utility as u

train, test = u.load_data()
train = u.prepare_data(train)
test = u.prepare_data(test)

# We want the Pclass, Age, Sex, Fare, SibSp, Parch, and Embarked variables
feature_list = ["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]
features_forest = train[feature_list].values
target = train["Survived"].values


# Building and fitting my_forest
forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 1)
my_forest = forest.fit(features_forest, target)

# Print the score of the fitted random forest
print(my_forest.score(features_forest, target))

# Compute predictions on our test set features then print the length of the prediction vector
test_features = test[feature_list].values
pred_forest = my_forest.predict(test_features)
print(len(pred_forest))

u.plot_result(feature_list, my_forest, features_forest, target)