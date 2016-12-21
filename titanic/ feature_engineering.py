# From:https://campus.datacamp.com
# Import the libraries
import utility as u

train, test = u.load_data()
train = u.prepare_data(train)

# Create train_two with the newly defined feature
train_two = train.copy()
train_two["Family_size"] = 1
train_two["Family_size"] = train_two["SibSp"] + train_two["Parch"] + 1
train_two["Family_size"] = train_two["Family_size"].fillna(1)
print(train_two)

# Create a new feature set and add the new feature
feature_list = ["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "Family_size"]

#Control overfitting by setting "max_depth" to 10 and "min_samples_split" to 5 : my_tree_two
max_depth = 10
min_samples_split = 5
my_tree, features, target = u.train_model(train_two, feature_list, max_depth, min_samples_split)

# Look at the importance and score of the included features
print(my_tree.feature_importances_)
print(my_tree.score(features, target))

u.plot_result(feature_list, my_tree, features, target)
