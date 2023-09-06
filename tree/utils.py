import numpy as np
import pandas as pd

def entropy(Y):
    cls = dict()
    for i in range(Y.size):
        if(Y.iat[i] in cls):
            cls[Y.iat[i]] += 1
        else:
            cls[Y.iat[i]] = 1
   
    entropy = 0
    for i in cls.keys():
        prob = cls[i]/Y.size
        entropy -= (prob*np.log2(prob))
   
    return entropy

def gini_index(Y):
    cls = dict()
    for i in range(Y.size):
        if(Y.iat[i] in cls):
            cls[Y.iat[i]] += 1
        else:
            cls[Y.iat[i]] = 1
   
    gini = 1


    for i in cls.keys():
        prob = cls[i]/Y.size
        gini -= np.square(prob)
   
    return gini

def information_gain(Y, attr):
    assert(attr.size==Y.size)


    cls_attr = dict()
    for i in range(attr.size):
        if(attr.iat[i] in cls_attr):
            cls_attr[attr.iat[i]].append(Y.iat[i])
        else:
            cls_attr[attr.iat[i]] = [Y.iat[i]]
   
    gain = entropy(Y)


    for i in cls_attr.keys():
        prob = len(cls_attr[i])/attr.size
        gain -= (prob*entropy(pd.Series(data=cls_attr[i])))
   
    return gain

