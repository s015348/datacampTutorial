# Read .csv file as list, array, and pandas object
import csv as csv
import numpy as np
import pandas as pd

csv_file_object = csv.reader(open('../titanic/train.csv', 'rb'))     # Load in the csv file
header = csv_file_object.next()                         # Skip the fist line as it is a header
train_list=[]                                                 # Create a variable to hold the data

for row in csv_file_object:                             # Skip through each row in the csv file,
    train_list.append(row[0:])                                 # adding each row to the data variable
print("\n\ntrain data in list:")
print(train_list)


train_array = np.array(train_list)                                     # Then convert from a list to an array.
print("\n\ntrain data in array:")
print(train_array)

train_dataframe = pd.read_csv('../titanic/train.csv')
print("\n\ntrain data in pandas dataframe:")
print(train_dataframe)
