import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statistics import mean

from metrics import *
from tree.base import DecisionTree


np.random.seed(42)

# Q(A)  .Read dataset
# ...
# 
from sklearn.datasets import make_classification

X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

# For plotting
import matplotlib.pyplot as plt

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.savefig("inputdata.png", dpi=150)

#making data set ready
# df = pd.DataFrame(X)
# categry = pd.DataFrame(y)
# df['category'] = categry

X = pd.DataFrame(X)
y = pd.Series(y,dtype="category")

split_percent = int(0.7*len(y))

#spliting to test and train sets
# ref https://towardsdatascience.com/how-to-split-a-dataset-into-training-and-testing-sets-b146b1649830
# training_data = df.sample(frac=0.7, random_state=25)
# testing_data = df.drop(training_data.index)

# Y_col = 'category'
# X_cols = df.loc[:, df.columns != Y_col].columns

# X_train = pd.DataFrame(training_data[X_cols])
# X_test = pd.DataFrame(testing_data[X_cols])
# y_train = pd.Series(training_data[Y_col],dtype="category" )
# y_test =  pd.Series(testing_data[Y_col],dtype="category" )

X_train = X.iloc[:split_percent, :]
X_test = X.iloc[split_percent:, :]
y_train = y.iloc[:split_percent]
y_test = y.iloc[split_percent:]

######
# Learning the classifier and testing 
for criteria in ['information_gain', 'gini_index']:
    tree = DecisionTree(criterion=criteria) 
    tree.fit(X_train, y_train)
    y_hat = tree.predict(X_test)
    # tree.plot()
    print('Criteria :', criteria)
    print('Accuracy: ', accuracy(y_hat, y_test))
    for cls in y.unique():
        print('Precision: ', precision(y_hat, y_test, cls))
        print('Recall: ', recall(y_hat, y_test, cls))



# #Q(2)#############

# # 5 fold cross-validation

# X = pd.DataFrame(X)
# y = pd.Series(y,dtype="category")

# assert (len(X[0]) == len(y))  # check for the length of dataframes
# temp_accurcy = 0                      # Initializes initial accuracy to 0
# for i in range(5):
#     split_start_index = int((i/5)*len(y))      #start index of split
#     split_end_index = int(((i+1)/5)*len(y))   #end index of split
#     X_trainn = pd.concat([X.iloc[:split_start_index, :], X.iloc[split_end_index:,:]], ignore_index=True)
#     X_testt = X.iloc[split_start_index:split_end_index, :]
#     y_trainn = pd.concat([y.iloc[:split_start_index], y.iloc[split_end_index:]], ignore_index=True)
#     y_testt = y.iloc[split_start_index:split_end_index]
#     tree = DecisionTree(criterion="information_gain", max_depth=3)
#     tree.fit(X_trainn, y_trainn)
#     y_test_hat = tree.predict(X_testt)
#     #print(type(y_test_hat), type(y_testt))
#     temp_accurcy += accuracy(y_test_hat, y_testt)

# print("5 fold cross-validation average accuracy:", temp_accurcy/5)


# # Q(2) part b 
# #Nested cross validation for optimal depth approximation



# def cross_validation(no_of_outer_folds:int,no_of_inner_folds:int,X:pd.DataFrame, y:pd.Series ) -> float:
#     temp_main_accuracy = []
#     for i in range(no_of_outer_folds):
#         split_start_index = int((i/no_of_outer_folds)*len(y))      #start index of split
#         split_end_index = int(((i+1)/no_of_outer_folds)*len(y))   #end index of split
#         X_train_outer = pd.concat([X.iloc[:split_start_index, :], X.iloc[split_end_index:,:]], ignore_index=True)
#         X_testt_outer = X.iloc[split_start_index:split_end_index, :]
#         y_train_outer = pd.concat([y.iloc[:split_start_index], y.iloc[split_end_index:]], ignore_index=True)
#         y_test_outer = y.iloc[split_start_index:split_end_index]

#         mx_acc = 0
#         max_acc_generating_depth = 0
#         for depth in range(10):
#             tmp_acc = 0
#             for j in range(no_of_inner_folds):
#                 validation_split_start = int((j/no_of_inner_folds)*X_train_outer.shape[0])
#                 validation_split_end = int(((j+1)/no_of_inner_folds)*X_train_outer.shape[0])
#                 X_train_inner = pd.concat([X.iloc[:validation_split_start, :], X.iloc[validation_split_end:,:]], ignore_index=True)
#                 X_validation = X.iloc[validation_split_start:validation_split_end, :]
#                 y_train_inner = pd.concat([y.iloc[:validation_split_start], y.iloc[validation_split_end:]], ignore_index=True)
#                 y_validation = y.iloc[validation_split_start:validation_split_end]

#                 tree = DecisionTree(criterion="information_gain", max_depth=depth)
#                 tree.fit(X_train_inner, y_train_inner)
#                 y_hat = tree.predict(X_validation)
#                 tmp_acc+= accuracy(y_hat, y_validation)
#             #find depth corrosponding to the maximum accuracy.
#             tmp_acc = tmp_acc / no_of_inner_folds #average of accuracy of inner validation folds
#             if(tmp_acc > mx_acc):
#                 mx_acc = tmp_acc
#                 max_acc_generating_depth = depth
#         tree = DecisionTree(criterion="information_gain", max_depth=max_acc_generating_depth)
#         tree.fit(X_train_outer,y_train_outer)
#         y_test_hat = tree.predict(X_testt_outer)
#         accur = accuracy(y_test_hat, y_test_outer)
#         temp_main_accuracy.append(accur)
#     return mean(temp_main_accuracy)

# X = pd.DataFrame(X)
# y = pd.Series(y,dtype="category")

# acc = cross_validation(5,7,X,y)
# print(acc)

