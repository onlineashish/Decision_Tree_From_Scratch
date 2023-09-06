
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
import seaborn as sns


np.random.seed(42)



'''
Create some fake data to do some experiments on the runtime complexity of your decision tree algorithm.
 Create a dataset with N samples and M binary features. Vary M and N to plot the time taken for:
  1) learning the tree, 2) predicting for test data. 
  How do these results compare with theoretical time complexity for decision tree creation and prediction. 
  You should do the comparison for all the four cases of decision trees. **[2 marks]**	
 >You should be editing `experiments.py` for the code containing the experiments.
'''
# Learn DTs 
# ...
# 
# Function to calculate average time (and std) taken by fit() and predict() for different N and P for 4 different cases of DTs
def averageTime(case):
    fit_times = []
    pred_times = []
    global fit_mesure
    fit_mesure= []
    global pred_mesure
    pred_mesure = []
    for N in range(50,250,50):
        for P in range(2,6,1):
            
            X, y = createFakeData(N,P,case)
            tree = DecisionTree(criterion="gini_index", max_depth=5)
            
            start_time = time.time()
            tree.fit(X,y)
            end_time = time.time()
            fit_times.append((N, P, end_time - start_time))
            #for performance measure
            if N == 200:
                fit_mesure.append(end_time - start_time)
            
            start_time = time.time()
            _ = tree.predict(X)
            end_time = time.time()
            pred_times.append((N, P, end_time - start_time))

            #for performance measure
            if N == 200:
                pred_mesure.append(end_time - start_time)


    fit_times=np.array(fit_times)
    pred_times=np.array(pred_times)
    plotTimings(fit_times,case+"fit")
    plotTimings(pred_times,case+"predict")
    return fit_times, pred_times
# Function to plot the results
# ..
# Function to create fake data (take inspiration from usage.py)
def createFakeData(N,P,case):
    
    if(case=="RR"):
        X = pd.DataFrame(np.random.randn(N, P))
        y = pd.Series(np.random.randn(N))
    elif(case=="RD"):
        X = pd.DataFrame(np.random.randn(N, P))
        y = pd.Series(np.random.randint(2, size = N), dtype="category")
    elif(case=="DD"):
        X = pd.DataFrame({i:pd.Series(np.random.randint(2, size = N), dtype="category") for i in range(P)})
        y = pd.Series(np.random.randint(2, size = N), dtype="category")
    else:
        X = pd.DataFrame({i:pd.Series(np.random.randint(2, size = N), dtype="category") for i in range(P)})
        y = pd.Series(np.random.randn(N))

    return X,y
# ...
# ..other functions

# def plotTimings(timeData):
#     df = pd.DataFrame(data=timeData)
#     heatmap1_data = pd.pivot_table(df, values='time', index=['N'], columns='P')
#     sns.heatmap(heatmap1_data, cmap="YlGnBu")
#     plt.show()

def plotTimings(times,case):
    
    plt.scatter(times[:, 0], times[:, 1], c=times[:, 2],cmap='viridis')
    plt.colorbar()
    plt.title((case))
    plt.xlabel('N')
    plt.ylabel('P')
    
    plt.savefig("./output_files/"+case+'.png')
    plt.clf()


def timeplot():
    x = [2,3,4,5]
    y1 = fit_mesure
    y2 = pred_mesure
    plt.figure(1)
    plt.plot(x, y1)
    plt.xlabel('P')
    plt.ylabel('time')
    plt.title("fit time")
    plt.savefig('runtime_compare_fit.png')
    
    plt.figure(2)
    plt.plot(x, y2)
    plt.xlabel('P')
    plt.ylabel('time')
    plt.title("predict time")
    plt.savefig('runtime_compare_pred.png')
    plt.clf()


#mnlog(n) runtime comparision
def runtime_compare_plot(lis_tuple,nam):
    new_data = np.zeros((lis_tuple.shape[0], 2))
    new_data[:, 0] = lis_tuple[:, 0] *  lis_tuple[:, 1] * np.log2(lis_tuple[:, 1])
    new_data[:, 1] = lis_tuple[:, 2]

    new_data = new_data[new_data[:, 0].argsort()]
    x = new_data[:, 0]
    y = new_data[:, 1]

    plt.plot(x, y)
    plt.xlabel('nmlog(m)')
    plt.ylabel('time')
    plt.title(nam)
    plt.savefig('./plots/'+nam+'.png')
    plt.clf()

def predict_classification_plot(lis_tuple,nam):
    new_data = np.zeros((lis_tuple.shape[0], 2))
    new_data[:, 0] = lis_tuple[:, 1] 
    new_data[:, 1] = lis_tuple[:, 2]

    new_data = new_data[new_data[:, 0].argsort()]
    x = new_data[:, 0]
    y = new_data[:, 1]

    plt.plot(x, y)
    plt.xlabel('M')
    plt.ylabel('time')
    plt.title(nam)
    plt.savefig('./plots/'+nam+'.png')
    plt.clf()

def predict_regression_plot(lis_tuple,nam):
    new_data = np.zeros((lis_tuple.shape[0], 2))
    new_data[:, 0] = lis_tuple[:, 0] 
    new_data[:, 1] = lis_tuple[:, 2]

    new_data = new_data[new_data[:, 0].argsort()]
    x = new_data[:, 0]
    y = new_data[:, 1]

    plt.plot(x, y)
    plt.xlabel('N')
    plt.ylabel('time')
    plt.title(nam)
    plt.savefig('./plots/'+nam+'.png')
    plt.clf()

    #storing fit time tuple and pred time tuple
fit_time_DR, pred_time_DR = averageTime("DR")
#timeplot()
runtime_compare_plot(fit_time_DR,"Fit DR")
predict_regression_plot(pred_time_DR,"Pred DR")

#timeplot can be created for all below cases
fit_time_DD, pred_time_DD = averageTime("DD")
runtime_compare_plot(fit_time_DD,"Fit DD")
predict_classification_plot(pred_time_DD,"Pred DD")

fit_time_RD, pred_time_RD = averageTime("RD")
runtime_compare_plot(fit_time_RD,"Fit RD")
predict_classification_plot(pred_time_DD,"Pred RD")

fit_time_RR, pred_time_RR = averageTime("RR")
runtime_compare_plot(fit_time_RR,"Fit RR")
predict_regression_plot(pred_time_RR,"Pred RR")
