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
filepath = '/Users/maycaj/Documents/HSI_III/Feb4_1x1_ContinuumRemoved_WLFiltered.csv' # '/Users/maycaj/Documents/HSI_III/1-22-25_5x5.csv' #'/Users/maycaj/Documents/HSI_III/Jan16_1x1_ContinuumRemoved.csv'
fracs = [0.045] # fraction of patches to include
iterations = 200 # how many times to repeat the analysis
threshold = 0.5 # what fraction of patches of edemaTrue to consider whole leg edemaTrue
nmStartEnd = ['Wavelength_451.18','Wavelength_954.83'] # 5x5: ['451.18','954.83'] # specify the wavelengths to include in analysis

data = pd.read_csv(filepath)
# data = data[data['final_diagnosis_other'] == 'Cellulitis'] # select the cellulitis condition only
data = data[data['ID'].isin([4,11,12,15,18,20,22,23,26,34,36])] # Select only cellulitis IDs

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
    minLen = int(min(dataTrue.shape[0],dataFalse.shape[0])) # find maximum number for each class
    trueSample = dataTrue.sample(n=minLen, random_state=random_state)
    falseSample = dataFalse.sample(n=minLen, random_state=random_state)
    underSampled = pd.concat([trueSample,falseSample], axis=0)
    return underSampled

def undersampleID(data):
    '''
    Evens out ID imbalances via undersampling
    Args:
        data: pandas dataframe
    Returns:
        undersampled dataframe
    '''
    underIDs = data['ID'].unique() # find unique IDs in data to undersample
    numIDs = []
    for ID in underIDs:
        numIDs.append(sum(data['ID'] == ID))
    minLen = min(numIDs)
    underSampled = pd.DataFrame([])
    for ID in underIDs:
        IDdata = data[data['ID'] == ID]
        IDsample = IDdata.sample(n=minLen, random_state=random_state)
        underSampled = pd.concat([underSampled,IDsample], axis=0)
    return underSampled

def weightClass(data):
    '''
    Evens out class inbalances via weighting
    Args:
        data: dataframe that contains classes to be weighted and a column named 'Weights'
    Returns:
        same dataframe with weights changed
    '''
    # classWeights = dataUnder[['Weights','Foldername']].groupby('Foldername').sum()
    # edemaTrue = classWeights.loc['EdemaTrue']
    # edemaFalse = classWeights.loc['EdemaFalse']

    # idWeights = dataUnder[('Weights','ID')].groupby('Foldername').sum()


    # for ID in uniqIDs:
    #     IDsum = sum(dataUnder.loc[dataUnder['ID']==ID,'Weights'])
    #     dataUnder.loc[dataUnder['ID']==ID,'Weights'] /= IDsum

    # weight by T/F 
    trueWeight = sum(data.loc[data['Foldername']=='EdemaTrue','Weights'])
    falseWeight = sum(data.loc[data['Foldername']=='EdemaFalse','Weights'])

    data.loc[data['Foldername']=='EdemaTrue','Weights'] /= trueWeight
    data.loc[data['Foldername']=='EdemaFalse','Weights'] /= falseWeight

    ## Make each ID and Foldername combination equally weighted
    # categorySum = dataUnder.groupby(['Foldername','ID']).size()
    # categorySum = categorySum.to_frame(name='Totals')
    # categorySum.reset_index(inplace=True)

    # for i in range(categorySum.shape[0]):
    #     folderName = categorySum.iloc[i,:]['Foldername']
    #     ID = categorySum.iloc[i,:]['ID']
    #     total = categorySum.iloc[i,:]['Totals']
    #     dataUnder.loc[(dataUnder['Foldername'] == folderName) | (dataUnder['ID'] == ID), 'Weights'] /= total


def printWeights(data):
    '''
    prints out the sums of the weights and the sums of the samples by Id and by T/F
    args: 
        data: dataframe 
    '''
    # Print out the sums of the weights for: 1) Each ID 2) T vs F
    for ID in uniqIDs:
        IDweight = sum(data.loc[data['ID'] == ID,'Weights'])
        print(f'{ID} weight sum: {IDweight}')

    trueWeight =  sum(data.loc[data['Foldername']=='EdemaTrue','Weights'])
    falseWeight = sum(data.loc[data['Foldername']=='EdemaFalse','Weights'])
    print(f'True Weight sum: {trueWeight}')
    print(f'False Weight sum: {falseWeight}')

    # print out numbers of samples
    for ID in uniqIDs:
        IDsum = sum(data['ID'] == ID)
        print(f'{ID} total: {IDsum}')
    trueSum = sum(data['Foldername']=='EdemaTrue')
    falseSum = sum(data['Foldername']=='EdemaFalse')
    print(f'True total: {trueSum}')
    print(f'False total: {falseSum}')


TN, FP, FN, TP = 0, 0, 0, 0 # initialize confusion matrix tally
uniqIDs = data['ID'].unique() # find unique IDs

IDaccs = pd.DataFrame(uniqIDs, columns=['ID']) # make a dataframe with accuracies for each ID
IDaccs = IDaccs.loc[IDaccs.index.repeat(2)]  # Repeat each row twice
IDaccs['Foldername'] = ['EdemaTrue', 'EdemaFalse'] * (len(IDaccs) // 2) # Add EdemaTrue and EdemaFalse to each row


# Loop over the selected number of data and each iteration
for selectedNum in selectedNums:
    for i in range(iterations): # iterate multiple times for error bars
        print(f'Starting iteration: {i}')
        random_state = np.random.randint(0,4294967295) # generate random integer for random state

        dataFrac = data.sample(n=selectedNum,random_state=random_state) # include a fraction of the data so debugging is fast

        # find a set of patient IDs that will go ONLY in testing or ONLY in training - a larger range makes it so that a wider range of ID+conditions are tested in less iterations
        train, test = splitByID(dataFrac, [0.08,0.4]) 

        # Undersample the majority class so there are equal numbers of each class
        train = undersampleClass(train, ['EdemaTrue','EdemaFalse'])

        # # Undersample by ID
        # train = undersampleID(train)

        # Add weights such that each patient and each category is equally weighted
        train['Weights'] = float(1)
        # # Weight classes by True / False
        # weightClass(train)
        # Xweights = train['Weights']

        # print sum of weights and sum of examples
        printWeights(train)

        # select the X and the y from the data
        Xtrain = train.loc[:,nmStartEnd[0]:nmStartEnd[1]] # leave out noisy wavelengths
        yTrain = train['Foldername']
        Xtest = test.loc[:,nmStartEnd[0]:nmStartEnd[1]]
        yTest = test['Foldername']
        
        # # Subtract each sample's mean from itself to get rid of flat PCA in first dimension (Subtracting mean worsens accuracy)
        # Xtrain = Xtrain.apply(lambda row: row-np.mean(row), axis=1)
        # Xtest = Xtest.apply(lambda row: row-np.mean(row), axis=1)

        # do PCA dimensionality reduction
        pca = make_pipeline(StandardScaler(), PCA(n_components=30, random_state=random_state))
        pca.fit(Xtrain, yTrain) # fit method's model

        # # Plot how much variance is explained by PCA
        # print(pca.named_steps['pca'].explained_variance_ratio_)
        # plt.figure(figsize=(8, 6))
        # plt.plot(range(1, len(pca.named_steps['pca'].explained_variance_ratio_) + 1), pca.named_steps['pca'].explained_variance_ratio_)
        # plt.xlabel('Number of Components')
        # plt.ylabel('Explained Variance Ratio')
        # plt.title(f'Scree Plot - Examples per iteration: {selectedNum}')
        # plt.show()

        # # plot the PCA weights
        # components = pca.named_steps['pca'].components_
        # components = np.abs(components)
        # wavelengths = [451.18,456.19,461.21,466.23,471.25,476.28,481.32,486.36,491.41,496.46,501.51,506.57,511.64,516.71,521.78,526.86,531.95,537.04,542.13,547.23,552.34,557.45,562.57,567.69,572.81,577.94,583.07,588.21,593.36,598.51,603.66,608.82,613.99,619.16,624.33,629.51,634.7,639.88,645.08,650.28,655.48,660.69,665.91,671.12,676.35,681.58,686.81,692.05,697.29,702.54,707.8,713.06,718.32,723.59,728.86,734.14,739.42,744.71,750.01,755.3,760.61,765.92,771.23,776.55,781.87,787.2,792.53,797.87,803.21,808.56,813.91,819.27,824.63,830,835.37,840.75,846.13,851.52,856.91,862.31,867.71,873.12,878.53,883.95,889.37,894.8,900.23,905.67,911.11,916.56,922.01,927.47,932.93,938.4,943.87,949.35,954.83]
        # for j in range(components.shape[0]):
        #     plt.scatter(wavelengths,components[j], label=f'Component: {j}')
        #     if j > 2: # don't plot more than 3 dimensions
        #       break   
        # plt.legend()
        # plt.title('PCA weights')
        # plt.xlabel('Band (nm)')
        # plt.ylabel('Absolute value of Weight')
        # plt.show()

        # fit SVM on data
        svc = SVC(kernel='linear')
        svc.fit(pca.transform(Xtrain),yTrain)
        # svc.fit(pca.transform(Xtrain),yTrain,sample_weight=Xweights) # for weights
        # svc.fit(Xtrain, yTrain) # for no dim reduction
        XtestTransformed = pca.transform(Xtest)
        # XtestTransformed = Xtest # for no dim reduction
        yPred = svc.predict(XtestTransformed)

        cm = confusion_matrix(yTest, yPred)
        print(cm)  # Output: [[1 1], [1 2]]
        TN = TN + cm[0, 0]  
        FP = FP + cm[0, 1] 
        FN = FN + cm[1, 0] 
        TP = TP + cm[1, 1] 
        patchAcc = (TN+TP)/(TN+TP+FN+FP)
        print(f'Confusion matrix for all iterations: \n [[{TP}, {FP}],\n [{FN}, {TN}]]')
        print(f'Patch Acc for all iterations: {patchAcc}')

        # find accuracy per ID
        test = test.copy() # make dataframe a copy instead of a view

        # add yTest and and yPred as columns of test
        test.loc[:,'correct'] = yTest == yPred # add a column that displays if answer is correct
        test.loc[:,'yPred'] = yPred
        test.loc[:,'yTest'] = yTest

        # Create a dataframe with patch accuracy - add to IDaccs
        print(f'Patch Acc this iteration: {np.mean(test["correct"])}')
        IDacc = test.groupby(['ID','Foldername'])['correct'].mean() # find average accuracy per ID
        IDacc = IDacc.to_frame(name=f'Iteration {i}') # convert to DataFrame
        IDacc.reset_index(inplace=True) # make 'ID' and 'Foldername' into separate columns
        IDaccs = IDaccs.merge(IDacc[['ID', 'Foldername', f'Iteration {i}']],on=['ID', 'Foldername'], how='left') # comment out when doing thresholding

        # # Create a dataframe with thresholding 
        # test['yPred Binary'] = test['yPred'].map({'EdemaTrue':True, 'EdemaFalse':False}) # map string labels to booleans
        # IDthresh = test.groupby(['ID','Foldername'])['yPred Binary'].mean() # find the fraction prediction of being true
        # IDthresh = IDthresh.to_frame(name='yPred Binary') # convert to DataFrame
        # IDthresh.reset_index(inplace=True) # make 'ID' and 'Foldername' into separate columns
        # IDthresh[f'Iteration {i}'] = ((IDthresh['Foldername'] == 'EdemaTrue') & (IDthresh['yPred Binary'] > threshold)) | ((IDthresh['Foldername'] == 'EdemaFalse') & (IDthresh['yPred Binary'] <= threshold)) # find bollean values for correctness
        # IDthresh[f'Iteration {i}'] = IDthresh[f'Iteration {i}'].astype(int) # convert boolean to integer
        # IDaccs = IDaccs.merge(IDthresh[['ID', 'Foldername', f'Iteration {i}']],on=['ID', 'Foldername'], how='left') # merge IDthresh values with the overall IDaccs

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
for thresh in [0.5]: # np.arange(0.3,0.7,0.1): # iterate over possible thresholds
    thresh = np.round(thresh, 1)
    IDroundAvg = IDaccs.drop(['ID','Foldername'], axis=1).T.apply(lambda cell: roundCells(cell,thresh)).mean() # round cells to threshold (thresh) and then take the mean across the rows
    IDaccs.insert(2,f'ID {thresh} Rounded Avg',IDroundAvg) # add thresholded values as a new column


def bootstrapRow(row, round=False):
    '''
    Bootstraps to make 95% CI across a pandas dataframe row
    Args:
        row: dataframe row
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
CI95round = iteration.apply(lambda row: bootstrapRow(row, round=True), axis=1)
IDaccs.insert(3, '95%CIround', CI95round)

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
yerrRounded = IDaccs.loc[:,'95%CIround'].apply(halfDifference)
IDaccs.insert(2, 'YerrRounded', yerrRounded)

# find values to plot in bar chart
IDaccs.insert(2,'Labels',IDaccs['ID'].astype(str) + '\n' + IDaccs['Foldername']) # combine columns for figure labels
IDaccs.loc['Column Average','Labels'] = 'Column Average'

# Find accuracy by Foldername; Add to IDaccs
FolderAcc = IDaccs.groupby('Foldername')['ID Avg'].mean() # find accuracy by foldername
FolderAcc = FolderAcc.reset_index(inplace=False)
FolderAcc['Labels'] = FolderAcc['Foldername']
IDaccs = pd.concat([IDaccs, FolderAcc], axis=0)

# Find rounded accuracy by Foldername; Add to IDaccs
FolderRound = IDaccs.groupby('Foldername')['ID 0.5 Rounded Avg'].mean()
FolderRound = FolderRound.reset_index(inplace=False)
IDaccs.loc[IDaccs['Labels']=='EdemaTrue','ID 0.5 Rounded Avg'] = FolderRound.loc[FolderRound['Foldername']=='EdemaTrue','ID 0.5 Rounded Avg']
IDaccs.loc[IDaccs['Labels']=='EdemaFalse','ID 0.5 Rounded Avg'] = FolderRound.loc[FolderRound['Foldername']=='EdemaFalse','ID 0.5 Rounded Avg']


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

columnBarPlot(IDaccs, IDaccs['ID 0.5 Rounded Avg'].notna(),'Labels','ID 0.5 Rounded Avg','YerrRounded','ID \n Foldername','Accuracy',f'0.5 Rounded Accuracy by ID Foldername n={selectedNum} iterations={iterations}')
columnBarPlot(IDaccs, IDaccs['ID Avg'].notna(),'Labels','ID Avg','Yerr','ID \n Foldername','Accuracy',f'Accuracy by ID Foldername n={selectedNum} iterations={iterations}')
plt.show()

# add metadata and save 
IDaccs.loc['Info','Labels'] = f'input filename: {filepath.split("/")[-1]}' # add input filename
IDaccs.to_csv(f'/Users/maycaj/Downloads/IDaccs_n={selectedNum}i={iterations}.csv') # Save the accuracy as csv
print('All done ;)')