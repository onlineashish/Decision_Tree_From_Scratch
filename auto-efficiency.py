
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from crossval import cross_validation

np.random.seed(42)
'''
Attribute Information:

1. mpg: continuous
2. cylinders: multi-valued discrete
3. displacement: continuous
4. horsepower: continuous
5. weight: continuous
6. acceleration: continuous
7. year: multi-valued discrete
8. origin: multi-valued discrete
9. car_name: string (unique for each instance)
'''

import pandas as pd
df = pd.read_fwf("./data/auto-mpg.data", names=["mpg", "cylinders" , "displacement", "horsepower", "weight",
 "acceleration", "year", "origin", "car_name" ])

#droping the rows where the value in column horsepower is ?
df = df.drop(df[df['horsepower'] == '?'].index)


#df.index = range(len(df))
df.reset_index(inplace=True, drop=True)

y = df['mpg']
df.drop(['car_name','mpg',], axis=1,inplace=True)

# training_data = df.sample(frac=0.7, random_state=42)
# testing_data = df.drop(training_data.index)
# X_train = training_data.iloc[:,1:]

X = pd.DataFrame(df)
y = pd.Series(y.values,dtype=float)
X[['displacement', 'horsepower','weight','acceleration']] = X[['displacement', 'horsepower','weight','acceleration']].astype(float)
X[['cylinders','year', 'origin']]= X[['cylinders','year', 'origin']].astype('category')


# accr = cross_validation(5,7,X,y)
# print('accuracy using cross validation:', accr)

# print(X.index)
# print(y.index)
# print(X.dtypes)
# print(y.dtype)

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

split_percent = int(0.6*len(y))
X_train = pd.DataFrame(X.iloc[:split_percent, :])
X_test = pd.DataFrame(X.iloc[split_percent:, :])
y_train = pd.Series(y.iloc[:split_percent].values)
y_test = pd.Series(y.iloc[split_percent:].values)

# for criteria in ['information_gain', 'gini_index']:
tree = DecisionTree(criterion='abc', max_depth=5) 
# tree.fit(X_train, y_train)
tree.fit(X, y)
y_hat = tree.predict(X_test)
#print('Criteria :', criteria)
print('RMSE: ', rmse(y_hat, y_test))
print('MAE: ', mae(y_hat, y_test))
# for cls in y_test.unique():
#     print('Precision: ', precision(y_hat, y_test, cls))
#     print('Recall: ', recall(y_hat, y_test, cls))

#inbuilt
print("sklearn decision tree regressor")
tree = DecisionTreeRegressor() 
tree.fit(X_train, y_train)
y_hat = tree.predict(X_test)
y_hat =pd.Series(y_hat)
print('InbuitRoot Mean Squared error: ', rmse(y_hat, y_test))
print('Inbuilt MAE: ', mae(y_hat, y_test))
# for cls in y_test.unique():
#     print('Precision: ', precision(y_hat, y_test, cls))
#     print('Recall: ', recall(y_hat, y_test, cls))

print('Score inbuilt: ', tree.score(X_test, y_test))
