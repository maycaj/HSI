import scipy.io
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
import itertools
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation
import re

# initialize filepath & number of samples
filepath = '/Users/maycaj/Documents/HSI_III/12-6-24_5x5.csv'
fracs = [0.01] # fraction of patches to include
iterations = 3 # how many times to repeat the analysis

data = pd.read_csv(filepath)
selectedNums = [int(frac*data.shape[0]) for frac in fracs] # convert the fraction to a number of examples


def splitByID(data, testRange):
    '''
    splits data into training and testing by ID number
    Args:
        data: Pandas Dataframe input
        testRange: list of [low,high] ranges for our testing dataset
    Returns:
        training and testing sets dataframes
    '''
    uniqIDs = data['ID'].unique()
    for i in range(300):
        random_state = np.random.randint(0,4294967295) # generate random integer for random state
        trainIDs, testIDs = train_test_split(uniqIDs, test_size=np.random.uniform(0.1,0.9), random_state=random_state) # split IDs randomly
        trainData = data[data['ID'].isin(trainIDs)]
        testData = data[data['ID'].isin(testIDs)]
        testSize = testData.shape[0] / (testData.shape[0] + trainData.shape[0])
        if testSize > testRange[0] and testSize < testRange[1]:
            print(f'Testsize split of {np.round(testSize, 3)} found after {i} iterations')
            # print(f'Train IDs: {trainIDs}')
            # print(f'Test IDs: {testIDs}')
            break
        if i == 99:
            print('Error: no test split found after 100 iterations')
    return trainData, testData

def undersampleClass(data, classes):
    '''
    Evens out class imbalances via undersampling
    Args:
        data: pandas dataframe 
        classes: string of binary classes to undersample ['class1','class2']
    Returns:
        undersampled data dataframe
    '''
    dataTrue = data[data['Foldername'] == classes[0]]
    dataFalse = data[data['Foldername'] == classes[1]]
    minLen = int(min(dataTrue.shape[0],dataFalse.shape[0], selectedNum/2)) # find maximum number for each class
    trueSample = dataTrue.sample(n=minLen, random_state=random_state)
    falseSample = dataFalse.sample(n=minLen, random_state=random_state)
    underSampled = pd.concat([trueSample,falseSample], axis=0)
    return underSampled

TN, FP, FN, TP = 0, 0, 0, 0 # initialize confusion matrix tally
uniqIDs = data['ID'].unique() # find unique IDs



IDaccs = pd.DataFrame(uniqIDs, columns=['ID']) # make a dataframe with accuracies for each ID
IDaccs = IDaccs.loc[IDaccs.index.repeat(2)]  # Repeat each row twice
IDaccs['Foldername'] = ['EdemaTrue', 'EdemaFalse'] * (len(IDaccs) // 2) # Add EdemaTrue and EdemaFalse to each row

for selectedNum in selectedNums:
    for i in range(iterations): # iterate multiple times for error bars
        print(f'Starting iteration: {i}')
        random_state = np.random.randint(0,4294967295) # generate random integer for random state
        
        # Undersample the majority class so there are equal numbers of each class
        dataUnder = undersampleClass(data, ['EdemaTrue','EdemaFalse'])

        # find a set of patient IDs that will go ONLY in testing or ONLY in training
        trainUnder, testUnder = splitByID(dataUnder, [0.08,0.13])

        # select the X and the y from the data
        Xtrain = trainUnder.loc[:,'451.18':'954.83'] # leave out noisy wavelengths
        yTrain = trainUnder['Foldername']
        Xtest = testUnder.loc[:,'451.18':'954.83']
        yTest = testUnder['Foldername']
        

        # do PCA dimensionality reduction
        pca = make_pipeline(StandardScaler(), PCA(n_components=3, random_state=random_state))
        pca.fit(Xtrain, yTrain) # fit method's model

        # fit SVM on data
        svc = SVC(kernel='linear')
        svc.fit(pca.transform(Xtrain),yTrain)
        XtestTransformed = pca.transform(Xtest)
        yPred = svc.predict(XtestTransformed)

        cm = confusion_matrix(yTest, yPred)
        print(cm)  # Output: [[1 1], [1 2]]
        TN = TN + cm[0, 0]  
        FP = FP + cm[0, 1] 
        FN = FN + cm[1, 0] 
        TP = TP + cm[1, 1] 
        patchAcc = (TN+TP)/(TN+TP+FN+FP)
        print(f'Patch Acc for all iterations: {patchAcc}')

        # find accuracy per ID
        testUnder = testUnder.copy() # make dataframe a copy instead of a view
        testUnder.loc[:,'correct'] = yTest == yPred # add a column that displays if answer is correct
        print(f'Patch Acc this iteration: {np.mean(testUnder["correct"])}')
        IDacc = testUnder.groupby(['ID','Foldername'])['correct'].mean() # find average accuracy per ID
        IDacc = IDacc.to_frame(name=f'Iteration {i}') # convert to DataFrame
        IDacc.reset_index(inplace=True) # make 'ID' and 'Foldername' into separate columns
        # IDaccs = pd.merge(IDaccs, IDacc, left_on=["('ID','Foldername')"], right_index=True, how='left') 

        IDaccs = IDaccs.merge(IDacc[['ID', 'Foldername', f'Iteration {i}']],on=['ID', 'Foldername'], how='left')


        pass

#Add averages and metadata to a new dataframe and save it
IDavg = IDaccs.select_dtypes(include=['number']).T.mean() #select the numerical data, and find mean across IDs using transpose
IDaccs.insert(1,'ID avg',IDavg) # Insert ID average
IDroundAvg = IDaccs.select_dtypes(include=['number']).T.round().mean() # same as above, but round first
IDaccs.insert(1,'ID Rounded Avg',IDroundAvg)
IDaccs.loc['Column Average'] = IDaccs.select_dtypes(include=['number']).mean(axis=0) # add the average for every iteration
IDaccs.loc['Info', IDaccs.columns[0]] = f'input filename: {filepath.split("/")[-1]}' # add input filename
IDaccs.to_csv(f'/Users/maycaj/Downloads/IDaccs.csv')
print('All done ;)')