# 4top_split.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 1. Import the data
x_file = './challenges/FOURTOPS/misc/X_dataset_CNN.csv'
y_file = './challenges/FOURTOPS/misc/Y_dataset_CNN.csv'

# Read CSV files (assuming they have headers; if not, add header=None)
df_X = pd.read_csv(x_file)
df_Y = pd.read_csv(y_file)

print("Original shapes:")
print("X shape:", df_X.shape)
print("Y shape:", df_Y.shape)

print("First 2 rows of X:")
print(df_X.head(2))
print("First 2 rows of Y:")
print(df_Y.head(2))

# Check if the number of rows matches
if df_X.shape[0] != df_Y.shape[0]:
    print("\nMismatch in number of rows detected.")
    # If one DataFrame has one extra row, check if it's empty and remove it
    if df_X.shape[0] > df_Y.shape[0]:
        if df_X.iloc[-1].isnull().all():
            print("Removing empty last row from X")
            df_X = df_X.iloc[:-1].reset_index(drop=True)
    elif df_Y.shape[0] > df_X.shape[0]:
        if df_Y.iloc[-1].isnull().all():
            print("Removing empty last row from Y")
            df_Y = df_Y.iloc[:-1].reset_index(drop=True)

# Assert the two DataFrames now have the same number of rows
assert df_X.shape[0] == df_Y.shape[0], "X and Y now have different numbers of rows!"
print("\nFinal number of events:", df_X.shape[0])

# 4. Split the data using stratification on the process ID.
stratify_col = df_X.iloc[:, 1]

# Split 80% training, 20% for validation + test.
X_train, X_temp, Y_train, Y_temp = train_test_split(
    df_X, df_Y, test_size=0.2, stratify=stratify_col, random_state=42
)

# Now split the remaining 20& into test and validation
stratify_temp = X_temp.iloc[:, 1]
X_val, X_test, Y_val, Y_test = train_test_split(
    X_temp, Y_temp, test_size=0.5, stratify=stratify_temp, random_state=42
)

print("\nSplit sizes:")
print("X_train:", X_train.shape[0])
print("X_val  :", X_val.shape[0])
print("X_test :", X_test.shape[0])
print("Total  :", X_train.shape[0] + X_val.shape[0] + X_test.shape[0])

# Optionally, print out the distribution of process IDs in each split
print("\nProcess ID distribution in training set:")
print(X_train.iloc[:, 1].value_counts())
print("\nProcess ID distribution in validation set:")
print(X_val.iloc[:, 1].value_counts())
print("\nProcess ID distribution in test set:")
print(X_test.iloc[:, 1].value_counts())

# 5. Remove the first THREE columns of the X set [PRIOR SET TO 2 AND AUC > .95]
X_train = X_train.iloc[:, 3:]
X_val   = X_val.iloc[:, 3:]
X_test = X_test.iloc[:, 3:]

# 6. Remove the last 13 columnts of the X set [THESE ARE ALL 0!!]
X_train = X_train.iloc[:, :-13]
X_val   = X_val.iloc[:, :-13]
X_test = X_test.iloc[:, :-13]

# 7. Check the dimensions of the sets after slicing
print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)

print("X_val shape:", X_val.shape)
print("Y_val shape:", Y_val.shape)

print("X_test shape:", X_test.shape)
print("Y_test shape:", Y_test.shape)

# 8. Save the splits to CSV files
X_train.to_csv('challenges/FOURTOPS/data/X_train.csv', index=False)
Y_train.to_csv('challenges/FOURTOPS/data//Y_train.csv', index=False)
X_val.to_csv('challenges/FOURTOPS/data/X_val.csv', index=False)
Y_val.to_csv('challenges/FOURTOPS/data/Y_val.csv', index=False)
X_test.to_csv('challenges/FOURTOPS/data/X_test.csv', index=False)
Y_test.to_csv('challenges/FOURTOPS/data//Y_test.csv', index=False)

print("\nSaved files: X_train.csv, Y_train.csv, X_val.csv, Y_val.csv, X_test.csv, Y_test.csv")