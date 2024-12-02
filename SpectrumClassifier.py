import scipy.io
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation
import re

# initialize filepath & number of samples
filepath = '/Users/maycaj/Documents/HSI_III/spectrums-12-1-24_EdemaTF_imgNum.csv'
selectedNums = [10,100,1000,10000]

X = np.genfromtxt(filepath, delimiter=',') #read in the .csv as a npy variable
X = X[1:,1:] # get rid of y column and column labels
y = pd.read_csv(filepath) # pull y column
y = y.iloc[:,0]
y_Categories = [item.split(' ')[0] for item in y] # pull EdemaTrue or EdemaFalse

accuracies = {}
for selectedNum in selectedNums:
    for i in range(50): # iterate multiple times for error bars
        print(f'SelectedNum: {selectedNum} Starting iteration: {i}')
        random_state = np.random.randint(0,4294967295) # generate random integer for random state

        # Select a subset of the original dataset
        testSize = 1 - (selectedNum / X.shape[0])
        X1, _, y1, _ = train_test_split(X, y, test_size=testSize, stratify=y_Categories, random_state=random_state)

        y_Categories1 = [item.split(' ')[0] for item in y1]
        y_int = [1 if item == 'EdemaTrue' else 0 for item in y_Categories1]
        y_int = np.array(y_int)

        # find a unique set of patient IDs that are either only in test OR only in training set
        IDs = [re.findall(r'\d+', item)[0] for item in y1]
        uniqIDs = list(set(IDs))
        trainIDs, testIDs = train_test_split(uniqIDs, test_size=0.3, random_state=random_state)

        # use IDs to find the IDXs of each example, and then split the data
        trainIDXs = []
        testIDXs = []
        for i, ID in enumerate(IDs):
            if ID in trainIDs:
                trainIDXs.append(i)
            elif ID in testIDs:
                testIDXs.append(i)
        X_train = X1[trainIDXs,:]
        X_test = X1[testIDXs,:]
        y_train = y_int[trainIDXs]
        y_test = y_int[testIDXs]

        nca = make_pipeline(StandardScaler(), NeighborhoodComponentsAnalysis(n_components=2, random_state=random_state))
        n_neighbors = 3
        knn = KNeighborsClassifier(n_neighbors=n_neighbors) # SVM and K-means also used commonly
        nca.fit(X_train, y_train) # fit method's model
        knn.fit(nca.transform(X_train), y_train) # compute the nearest neighbor classifier on the transformed training set
        acc_knn = knn.score(nca.transform(X_test), y_test) # compute the nearest neighbor accuracy on the transformed test set=
        knn.fit(nca.transform(X_train), y_train) # compute the nearest neighbor classifier on the transformed training set
        plt.figure()
        X_embedded = nca.transform(X1) # transform (embed) the entire dataset 
        scatter = plt.scatter(X_embedded[:,0],X_embedded[:,1],c=y_int,s=15,cmap="Set1", alpha=0.4, marker='.')     # (x, y, color, size, colormap) c=targets if we want a different color for each target
        n_examples = X_train.shape[0]
        plt.title("\n n_train={}, KNN (k={} Test accuracy = {:.2f})".format(n_examples, n_neighbors, acc_knn))
        plt.xlabel('Component 1')
        plt.ylabel('Compoenent 2')
        # plt.show()

        key = str(selectedNum)
        if key not in accuracies:
            accuracies[key] = []
        accuracies[key].append(acc_knn)
        print('')

def getErrorBars(input): # do power analysis of 1D array to get 95% CI using bootstrapping
    input = np.array(input)
    avgs = np.array([])
    for _ in range(1000): # randomly choose a subset of the data with replacement and find the average 
        choices = np.random.choice(input, size=input.shape[0], replace=True)
        avg = np.mean(choices)
        avgs = np.append(avgs, avg)
    inputAvg = np.mean(input) # average 
    STDEV = np.std(avgs) # find the standard deviation across each of the averages
    marginOfError = (1.96*STDEV)/np.sqrt(len(input)) #z*=1.96 for a 95% confidence interval; marginOfError = ((z*)*STDEV)/sqrt(numSamples)
    CI95 = np.array([inputAvg-marginOfError, inputAvg+marginOfError]) # confidence interval = Average +- marginOfError
    return marginOfError, inputAvg

def accDict2chart(accuracies, exptNum): # make a bar chart with error bars from a dictionary with the accuracies for each start_ends
    marginOfErrors = []
    inputAvgs = []
    for key in accuracies: # key is the start_ends 
        epochAcc = accuracies[key] # epochAcc is a list of accuracy for each start_ends
        marginOfError, inputAvg = getErrorBars(epochAcc) # find error bars for each start_ends
        marginOfErrors.append(marginOfError)
        inputAvgs.append(inputAvg) 
    keys = accuracies.keys()
    fig, ax = plt.subplots()
    bars = ax.bar(keys, inputAvgs, yerr=marginOfErrors, capsize=5, color='skyblue', edgecolor='black') # plot bar chart with average accuracy and error bars
    ax.set_xlabel('Number of training and testing spectra')
    ax.set_ylabel('Accuracy')
    title = exptNum + 'Accuracy per epoch with 95% CI'
    ax.set_title(title)
accDict2chart(accuracies, '')
plt.show()

print('All done!')