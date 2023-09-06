import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statistics import mean

from metrics import *
from tree.base import DecisionTree

def cross_validation(no_of_outer_folds:int,no_of_inner_folds:int,X:pd.DataFrame, y:pd.Series ) -> float:
    temp_main_accuracy = []
    for i in range(no_of_outer_folds):
        split_start_index = int((i/no_of_outer_folds)*len(y))      #start index of split
        split_end_index = int(((i+1)/no_of_outer_folds)*len(y))   #end index of split
        X_train_outer = pd.concat([X.iloc[:split_start_index, :], X.iloc[split_end_index:,:]], ignore_index=True)
        X_testt_outer = X.iloc[split_start_index:split_end_index, :]
        y_train_outer = pd.concat([y.iloc[:split_start_index], y.iloc[split_end_index:]], ignore_index=True)
        y_test_outer = y.iloc[split_start_index:split_end_index]

        mx_acc = 0
        max_acc_generating_depth = 0
        for depth in range(10):
            tmp_acc = 0
            for j in range(no_of_inner_folds):
                validation_split_start = int((j/no_of_inner_folds)*X_train_outer.shape[0])
                validation_split_end = int(((j+1)/no_of_inner_folds)*X_train_outer.shape[0])
                X_train_inner = pd.concat([X.iloc[:validation_split_start, :], X.iloc[validation_split_end:,:]], ignore_index=True)
                X_validation = X.iloc[validation_split_start:validation_split_end, :]
                y_train_inner = pd.concat([y.iloc[:validation_split_start], y.iloc[validation_split_end:]], ignore_index=True)
                y_validation = y.iloc[validation_split_start:validation_split_end]

                tree = DecisionTree(criterion="information_gain", max_depth=depth)
                tree.fit(X_train_inner, y_train_inner)
                y_hat = tree.predict(X_validation)
                tmp_acc+= accuracy(y_hat, y_validation)
            #find depth corrosponding to the maximum accuracy.
            tmp_acc = tmp_acc / no_of_inner_folds #average of accuracy of inner validation folds
            if(tmp_acc > mx_acc):
                mx_acc = tmp_acc
                max_acc_generating_depth = depth
        tree = DecisionTree(criterion="information_gain", max_depth=max_acc_generating_depth)
        tree.fit(X_train_outer,y_train_outer)
        y_test_hat = tree.predict(X_testt_outer)
        accur = accuracy(y_test_hat, y_test_outer)
        temp_main_accuracy.append(accur)
    return mean(temp_main_accuracy)