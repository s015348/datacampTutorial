# From:https://campus.datacamp.com
# Import the libraries
import utility as u

# Get the model
train, test = u.load_data()
train = u.prepare_data(train)

# Create a new array with the added features: features_two
#print(train.describe())
feature_list = ["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "PassengerId"]

#Control overfitting by setting "max_depth" to 10 and "min_samples_split" to 5 : my_tree_two
max_depth = 10
min_samples_split = 5
my_tree, features, target = u.train_model(train, feature_list, max_depth, min_samples_split)


#Print the score of the new decison tree
# Look at the importance and score of the included features
print(my_tree.feature_importances_)
print(my_tree.score(features, target))

u.plot_result(feature_list, my_tree, features, target)