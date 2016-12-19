# From:https://campus.datacamp.com
# Import the Pandas library
import pandas as pd

# Load the train and test datasets to create two DataFrames
train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
train = pd.read_csv(train_url)

test_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"
test = pd.read_csv(test_url)

# absolute numbers
survived_num = train["Survived"].value_counts()

# percentages
survived_perc = train["Survived"].value_counts(normalize = True)

print("Total survived:")
print(survived_num)
print(survived_perc)

male_survived_num = train["Survived"][train["Sex"] == 'male'].value_counts()
female_survived_num = train["Survived"][train["Sex"] == 'female'].value_counts()
male_survived_perc = train["Survived"][train["Sex"] == 'male'].value_counts(normalize = True)
female_survived_perc = train["Survived"][train["Sex"] == 'female'].value_counts(normalize = True)

print("Male survived:")
print(male_survived_num)
print(male_survived_perc)

print("Female survived:")
print(female_survived_num)
print(female_survived_perc)