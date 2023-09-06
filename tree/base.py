"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .utils import  information_gain, gini_index


np.random.seed(42)

# creating class for a tree node
class TreeNode():
    def __init__(self):
        
        self.isCategory_type = False
        self.feature = None
        self.child = dict()
        self.isLeaf = False
        self.splitValue = None
        self.value = None


class DecisionTree():
    def __init__(self, criterion, max_depth=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.root = None


   
    def create_tree(self,X,y,currdepth):

        currnode = TreeNode()   # Creating a new Tree Node

        feature = -1
        split_value = None
        best_measure = None


        #Gini Gain calculation function
        def gain_by_gini(xcol,y,splitValue):
            if (splitValue==None):
                classes1 = np.unique(xcol)
                s = 0
                for j in classes1:
                    y_sub=[]
                    for k in range(y.size):
                        if xcol[k]==j:
                            y_sub.append(y[k])
                    y_sub=pd.Series(y_sub)
                   
                    s += y_sub.size*gini_index(y_sub)
                ans = -1*(s/xcol.size)
                return ans
            else:
                
                y_sub1=[]
                for k in range(y.size):
                    if xcol[k]<=splitValue:
                        y_sub1.append(y[k])
                y_sub1=pd.Series(y_sub1)
               
                
                y_sub2=[]
                for k in range(y.size):
                    if xcol[k]>splitValue:
                        y_sub2.append(y[k])
                y_sub2=pd.Series(y_sub2,dtype=y.dtype)
                ans = y_sub1.size*gini_index(y_sub1) + y_sub2.size*gini_index(y_sub2)
                ans =  -1*(ans/y.size)
                return ans
        def loss_fun(xcol,y,splitValue):
            if(xcol.dtype.name=="category"):
                classes1 = np.unique(xcol)
                ans = 0
                for j in classes1:
                    
                    y_sub=[]
                    for k in range(y.size):
                        if xcol[k]==j:
                            y_sub.append(y[k])
                    y_sub=pd.Series(y_sub)
                    ans += y_sub.size*np.var(y_sub)
            else:
                
                y_sub1=[]
                for k in range(y.size):
                    if xcol[k]<=splitValue:
                        y_sub1.append(y[k])
                y_sub1=pd.Series(y_sub1)
                            
                            
                y_sub2=[]
                for k in range(y.size):
                    if xcol[k]>splitValue:
                        y_sub2.append(y[k])
                y_sub2=pd.Series(y_sub2,dtype=y.dtype)            
                ans = y_sub1.size*np.var(y_sub1) + y_sub2.size*np.var(y_sub2)
            return ans

        # Classification Problems
        if(y.dtype.name=="category"):
            classes = np.unique(y)
            #base condition for recursion
            if(classes.size==1 or (self.max_depth!=None and self.max_depth==currdepth)or X.shape[1]==0):
                currnode.isLeaf = True          # make it leaf if either all value of y is same
                currnode.isCategory_type = True  # or we have reached max deft or there is only one atribut
                if(classes.size==1):
                    currnode.value = classes[0]
                else:
                    currnode.value = y.value_counts().idxmax()


                return currnode
             


            for i in X:
                xcol = X[i]
               
                # Discreate Input and Discreate Output
                if(xcol.dtype.name=="category"):
                    measure = None
                    if(self.criterion=="information_gain"):         # Criteria is Information Gain
                        measure = information_gain(y,xcol)
                    else:
                        split_value = None                                           # Criteria is Gini Index
                        measure = gain_by_gini(xcol,y,split_value)
                    if(best_measure==None or best_measure<measure ):
                        feature = i
                        best_measure = measure
                        split_value = None


                # Real Input and Discreate Output
                else:
                    xcol_sorted = xcol.sort_values()
                    for j in range(xcol_sorted.size-1):
                        index = xcol_sorted.index[j]
                        next_index = xcol_sorted.index[j+1]
                        if(y[index]!=y[next_index]):
                            measure = None
                            splitValue =np.mean([xcol[index],xcol[next_index]])
                           
                            if(self.criterion=="information_gain"):                 # Criteria is Information Gain
                                helper_attr = pd.Series(xcol<=splitValue)
                                measure = information_gain(y,helper_attr)
                           
                            else:                                                   # Criteria is Gini Index
                                measure=gain_by_gini(xcol,y,splitValue)
                            if(best_measure==None or best_measure<measure ):
                                feature = i               # update make when getting better split or it's first split
                                best_measure = measure
                                split_value = splitValue


                         
           
       
        # Regression Problems
         
                    
        else:
            

            #base case for recursion
            if( (self.max_depth!=None and self.max_depth==currdepth) or y.size==1 or X.shape[1]==0):
                currnode.isLeaf = True# make it leaf if reched to max deft or there is only one row
                if(y.size==1):
                    currnode.value = y[0]  # this about this
                else:
                    currnode.value = y.mean()
                return currnode
        
            for i in X:
                xcol = X[i]


                # Discreate Input Real Output
                if(xcol.dtype.name=="category"):
                    splitValue=None
                    measure=loss_fun(xcol,y,splitValue)
                    if(best_measure==None or best_measure>measure):#gain update if it's first  
                        best_measure = measure                     # or it's better gain
                        feature = i
                        split_value = None


                # Real Input Real Output
                else:
                    xcol_sorted = xcol.sort_values()
                    for j in range(y.size-1):
                        index = xcol_sorted.index[j]
                        next_index = xcol_sorted.index[j+1]
                        splitValue =np.mean([xcol[index],xcol[next_index]])
                       
                        measure=loss_fun(xcol,y,splitValue)
                        
                  
                        if(best_measure==None or best_measure>measure):# made update when splite is first
                            feature = i                               # or geting better splite
                            best_measure = measure
                            split_value = splitValue


        # when current treenode is category based
        if(split_value==None):
            currnode.isCategory_type = True
            currnode.feature = feature
            classes = np.unique(X[feature])
            for j in classes:
                #y_new = pd.Series([y[k] for k in range(y.size) if X[feature][k]==j], dtype=y.dtype)
                y_new=[]
                for k in range(y.size):
                    if X[feature][k]==j:
                      y_new.append(y[k])
                y_new=pd.Series(y_new,dtype=y.dtype)

                X_new = X[X[feature]==j].reset_index().drop(['index',feature],axis=1)
                currnode.child[j] = self.create_tree(X_new, y_new, currdepth+1)

        
        else:# when current treenode is split based
            currnode.feature = feature
            currnode.splitValue = split_value
            y_new1 = pd.Series([y[k] for k in range(y.size) if X[feature][k]<=split_value], dtype=y.dtype)
            X_new1 = X[X[feature]<=split_value].reset_index().drop(['index'],axis=1)
            y_new2 = pd.Series([y[k] for k in range(y.size) if X[feature][k]>split_value], dtype=y.dtype)
            X_new2 = X[X[feature]>split_value].reset_index().drop(['index'],axis=1)
            currnode.child["lt"] = self.create_tree(X_new1, y_new1, currdepth+1)
            currnode.child["gt"] = self.create_tree(X_new2, y_new2, currdepth+1)
       
        return currnode




    def fit(self, X, y):
        assert(y.size>0)
        assert(X.shape[0]==y.size)
        self.root = self.create_tree(X,y,0)


    def predict(self, X):
        y_hat = list()                  # List to contain the predicted values


        for i in range(X.shape[0]):
            xrow = X.iloc[i,:]          # Get an instance of the data for prediction purpose


            h = self.root
            while(not h.isLeaf):                            # when treenode is not a leaf
                if(h.isCategory_type):                       # when treenode is category based
                    h = h.child[xrow[h.feature]]
                else:                                       # when treenode is split based
                    if(xrow[h.feature]<=h.splitValue):
                        h = h.child["lt"]
                    else:
                        h = h.child["gt"]
           
            y_hat.append(h.value)                           #when treenode is a leaf
       
        y_hat = pd.Series(y_hat)


        return y_hat




    def plotTree(self, root, depth):
        if(root.isLeaf):
            if(root.isCategory_type):
                return "Class "+str(root.value)
            else:
                return "Value "+str(root.value)


        s = ""
        if(root.isCategory_type):
            for i in root.child.keys():
                s += "?("+str(root.feature)+" == "+str(i)+")\n"
                s += "\t"*(depth+1)
                s += str(self.plotTree(root.child[i], depth+1)).rstrip("\n") + "\n"
                s += "\t"*(depth)
            s = s.rstrip("\t")
        else:
            s += "?("+str(root.feature)+" <= "+str(root.splitValue)+")\n"
            s += "\t"*(depth+1)
            s += "Y: " + str(self.plotTree(root.child["lt"], depth+1)).rstrip("\n") + "\n"
            s += "\t"*(depth+1)
            s += "N: " + str(self.plotTree(root.child["gt"], depth+1)).rstrip("\n") + "\n"
       
        return s
           


    def plot(self):
        h = self.root
        s = self.plotTree(h,0)
        print(s)


