from typing import Union
import pandas as pd


def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """

    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert y_hat.size == y.size
    assert y.size > 0
    count = 0
    for i in range(y_hat.size):
        if(y_hat.iat[i]==y.iat[i]):
            count+=1
    
    return count/y.size


def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """
    assert y_hat.size == y.size
    TP = 0
    TP_and_FP = 0
    for i in range(y_hat.size):
        if(y_hat.iat[i]==cls):
            if(y_hat.iat[i]==y.iat[i]):
                TP+=1
            TP_and_FP+=1
    if(TP == 0): 
        return 0
    return TP/TP_and_FP


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """
    assert(y_hat.size == y.size)
    TP = 0
    TP_and_TN = 0
    for i in range(y_hat.size):
        if(y.iat[i]==cls):
            if(y_hat.iat[i]==y.iat[i]):
                TP+=1
            TP_and_TN+=1

    if(TP == 0):
        return 0
    return TP/TP_and_TN


def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """
    assert(y_hat.size == y.size)
    assert(y_hat.size > 0)
    return ((y-y_hat)**2).mean()**0.5
    


def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """
    assert(y_hat.size == y.size)
    assert(y_hat.size > 0)
    return abs(y-y_hat).mean()
