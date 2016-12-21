# From:https://campus.datacamp.com
# Import the libraries
import numpy as np
import pandas as pd
import utility as u

# Get the model
train, test = u.load_data()
train = u.prepare_data(train)
feature_list = ["Pclass", "Sex", "Age", "Fare"]
my_tree, features, target = u.train_model(train, feature_list)

# Clean test data
test = u.prepare_data(test)
# Impute the missing value with the median
#print(test.Fare.describe())
test["Fare"] = test["Fare"].fillna(test["Fare"].median())
#print(test.Fare.describe())

my_prediction = u.predict_result(test, feature_list, my_tree)

# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
PassengerId =np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])
print(my_solution.describe())


# Check that your data frame has 418 entries
print(my_solution.shape)

# Write your solution to a csv file with the name my_solution.csv
my_solution.to_csv("my_solution_one.csv", index_label = ["PassengerId"])