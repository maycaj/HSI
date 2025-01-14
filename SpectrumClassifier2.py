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
filepath = '/Users/maycaj/Documents/HSI_III/1-7-25_5x5.csv'
fracs = [0.01] # fraction of patches to include
iterations = 15 # how many times to repeat the analysis
threshold = 0.7 # what fraction of patches of edemaTrue to consider whole leg edemaTrue

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
    for i in range(1000):
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

        # add yTest and and yPred as columns of testUnder
        testUnder.loc[:,'correct'] = yTest == yPred # add a column that displays if answer is correct
        testUnder.loc[:,'yPred'] = yPred
        testUnder.loc[:,'yTest'] = yTest

        # Create a dataframe with patch accuracy - add to IDaccs
        print(f'Patch Acc this iteration: {np.mean(testUnder["correct"])}')
        IDacc = testUnder.groupby(['ID','Foldername'])['correct'].mean() # find average accuracy per ID
        IDacc = IDacc.to_frame(name=f'Iteration {i}') # convert to DataFrame
        IDacc.reset_index(inplace=True) # make 'ID' and 'Foldername' into separate columns
        # IDaccs = IDaccs.merge(IDacc[['ID', 'Foldername', f'Iteration {i}']],on=['ID', 'Foldername'], how='left') # comment out when doing thresholding

        # Create a dataframe with thresholding 
        testUnder['yPred Binary'] = testUnder['yPred'].map({'EdemaTrue':True, 'EdemaFalse':False}) # map string labels to booleans
        IDthresh = testUnder.groupby(['ID','Foldername'])['yPred Binary'].mean() # find the fraction prediction of being true
        IDthresh = IDthresh.to_frame(name='yPred Binary') # convert to DataFrame
        IDthresh.reset_index(inplace=True) # make 'ID' and 'Foldername' into separate columns
        IDthresh[f'Iteration {i}'] = ((IDthresh['Foldername'] == 'EdemaTrue') & (IDthresh['yPred Binary'] > threshold)) | ((IDthresh['Foldername'] == 'EdemaFalse') & (IDthresh['yPred Binary'] <= threshold)) # find bollean values for correctness
        IDthresh[f'Iteration {i}'] = IDthresh[f'Iteration {i}'].astype(int) # convert boolean to integer
        IDaccs = IDaccs.merge(IDthresh[['ID', 'Foldername', f'Iteration {i}']],on=['ID', 'Foldername'], how='left') # merge IDthresh values with the overall IDaccs

        pass

def roundCells(cell, thresh):
    '''
    rounds cells to desired threshold
    args:
        cell: cell of a dataframe
        thresh: threshold decimal
    returns:
        output: rounded cells
    '''
    output =  np.where(np.isnan(cell), np.nan, np.where(cell >= thresh, 1, 0))
    return output

#Add averages to dataframe
IDavg = IDaccs.drop(['ID','Foldername'], axis=1).T.mean() #Exclude the non-numerical data, and find mean across IDs using transpose
IDaccs.insert(2,'ID Avg',IDavg) # Insert ID average
# for thresh in [0.5]: # np.arange(0.3,0.7,0.1): # iterate over possible thresholds
#     thresh = np.round(thresh, 1)
#     IDroundAvg = IDaccs.drop(['ID','Foldername'], axis=1).T.apply(lambda cell: roundCells(cell,thresh)).mean() # round cells to threshold (thresh) and then take the mean across the rows
#     IDaccs.insert(2,f'ID {thresh} Rounded Avg',IDroundAvg) # add thresholded values as a new column


def bootstrapRow(row, round=False):
    '''
    Bootstraps to make 95% CI across a pandas dataframe row
    Args:
        row: dataframe rowq
    Returns:
        lower: lower end of CI
        upper: higher end of CI
    '''
    confidence = 0.95
    bootstrapped_means = []
    row = row[row.notna()].values # remove nan values
    if round:
        row = np.round(row)
    for _ in range(100): #___ set to 1000 for deployment
        sample = np.random.choice(row, size=len(row), replace=True)
        bootstrapped_means.append(np.mean(sample))
    lower = np.percentile(bootstrapped_means, (1-confidence)*100/2)
    upper = np.percentile(bootstrapped_means, (1+confidence)*100/2)
    upperLower = np.array([np.round(float(lower),2), np.round(float(upper),2)])
    return upperLower

# add average across columns
IDaccs.loc['Column Average'] = IDaccs.drop(['ID','Foldername'], axis=1).mean(axis=0) 

# apply 95% CI to iterations data
hasFolderHasAvg = (IDaccs.loc[:,'Foldername'].notna()) | (IDaccs.index == 'Column Average') # select rows with folder names and the column average
iteration = IDaccs.loc[hasFolderHasAvg, 'Iteration 0':f'Iteration {i}'] # select all of iteration data
CI95 = iteration.apply(lambda row: bootstrapRow(row), axis=1)
IDaccs.insert(3, '95%CI', CI95)
# CI95round = iteration.apply(lambda row: bootstrapRow(row, round=True), axis=1)
# IDaccs.insert(3, '95%CIround', CI95round)

def halfDifference(input):
    '''
    construct yerror 
    input: [lowerCI, upperCI]
    output: half of difference between lower CI and upper CI
    '''
    return np.abs(input[0]-input[1])/2
# notColAvg = IDaccs.index != 'Column Average'
yerr = IDaccs.loc[:,'95%CI'].apply(halfDifference) # apply the halfDifference equation to get the error needed for bar chart
IDaccs.insert(2, 'Yerr', yerr)
# yerrRounded = IDaccs.loc[:,'95%CIround'].apply(halfDifference)
# IDaccs.insert(2, 'YerrRounded', yerrRounded)

# find values to plot in bar chart
IDaccs.insert(2,'Labels',IDaccs['ID'].astype(str) + '\n' + IDaccs['Foldername']) # combine columns for figure labels
IDaccs.loc['Column Average','Labels'] = 'Column Average'

# Find accuracy by Foldername
FolderAcc = IDaccs.groupby('Foldername')['ID Avg'].mean() # find accuracy by foldername
FolderAcc = FolderAcc.reset_index(inplace=False)
FolderAcc['Labels'] = FolderAcc['Foldername']
IDaccs = pd.concat([IDaccs, FolderAcc], axis=0)

def columnBarPlot(df, hasValues, labelsKey, valuesKey, errorKey, xlabel, ylabel, title):
    '''
    plots down a column
    creates bar chart with error bars
    args:
        df: dataFrame
        hasValues: Key for Dataframe column. The column's rows with values will be used in this analysis
        labelsKey: Key for labels column in Dataframe
        valuesKey: Key for values column in Dataframe
        xlabel, ylabel, title: for plot
    '''
    labels = df[hasValues][labelsKey] # labels for bar chart
    labels.loc['Column Average'] = 'Column Average' # add label for column average
    values = df[hasValues][valuesKey] # y values for bar chart
    yerrPlot = df[hasValues][errorKey]
    yerrPlot = np.vstack(yerrPlot).T # errors for bar chart
    # plot bar chart
    fig, ax = plt.subplots()
    bars = ax.bar(labels, values, yerr=yerrPlot)
    ax.tick_params(axis='x', labelsize=3)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title + ' overall average=' + str(round(values['Column Average']*100,1))) 

# columnBarPlot(IDaccs, IDaccs['ID 0.5 Rounded Avg'].notna(),'Labels','ID 0.5 Rounded Avg','YerrRounded','ID \n Foldername','Accuracy',f'0.5 Rounded Accuracy by ID Foldername n={selectedNum} iterations={iterations}')
columnBarPlot(IDaccs, IDaccs['ID Avg'].notna(),'Labels','ID Avg','Yerr','ID \n Foldername','Accuracy',f'Accuracy by ID Foldername n={selectedNum} iterations={iterations}')
plt.show()

# add metadata and save 
IDaccs.loc['Info','Labels'] = f'input filename: {filepath.split("/")[-1]}' # add input filename
IDaccs.to_csv(f'/Users/maycaj/Downloads/IDaccs_n={selectedNum}i={iterations}.csv') # Save the accuracy as csv
print('All done ;)')