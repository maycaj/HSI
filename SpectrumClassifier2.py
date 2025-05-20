#!/Users/maycaj/Documents/HSI_III/.venv/bin/python3

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import SGDClassifier
import time
from joblib import Parallel, delayed
from HSI_Functions import convolve_spectra


## INITIALIZE PARAMETERS
filepath = '/Users/maycaj/Documents/HSI_III/PatchCSVs/Mar25_NoCR_PostDeoxyCrop.csv' #'/Users/maycaj/Documents/HSI_III/PatchCSVs/Feb4_1x1_ContinuumRemoved_WLFiltered.csv' #'/Users/maycaj/Documents/HSI_III/Apr_8_CR_FullRound1and2.csv'  
fracs = [0.01] # fraction of patches to include
# filepath = '/Users/maycaj/Documents/HSI_III/PatchCSVs/DebugData.csv' # for fast debugging
# fracs = [1]
iterations = 3 # how many times to repeat the analysis
threshold = 0.5 # what fraction of patches of edemaTrue to consider whole leg edemaTrue
nmStart = 'Wavelength_451.18'
nmEnd = 'Wavelength_954.83'
n_jobs = -1 # Number of CPUs to use during each Fold. -1 = as many as possible
stochastic = False
y_col = 'Foldername' 

# Load DataFrame
start_time = time.time()
print('Loading dataframe...')
data = pd.read_csv(filepath)
print(f'Done loading dataframe ({np.round(time.time()-start_time,1)}s)')
y_categories = data[y_col].unique() # ['Male','Female'] # [0,1] is [short, tall] and [pale, dark]

## PROCESS DATA FOR CERTAIN y
data = data[(data['ID'].isin([11,12,15,18,20,22,23,26,34,36])) | (data['Foldername'] =='EdemaFalse')] # Round 1: Select only cellulitis IDs or IDs in EdemaFalse Folder
# data = data[(data['ID'].isin([1,2,5,6,7,8,9,10,13,14,19,21,24,27,29,30,31,32,33,35,37,38,39,40]) | (data['Foldername'] =='EdemaFalse'))] # Round 1: Select Peripheral IDs or IDs in EdemaFalse folder
# data = data[(data['ID'].isin([11,12,15,18,20,22,23,26,34,36,45,59,61,70])) | (data['Foldername'] =='EdemaFalse')] # Round 2: Select only cellulitis IDs or IDs in EdemaFalse Folder
data = data[data['ID'] != 0] # Remove Dr. Pare

n_splits = len(data['ID'].unique()) # number of splits to make the fold
col_names = list(data.loc[:,nmStart:nmEnd].columns)

# Convolve the skin spectra with the chemophore spectras
HbO2Path = '/Users/maycaj/Documents/HSI_III/Absorbances/HbO2 Absorbance.csv'
convolve_spectra(data, [nmStart,nmEnd], 'HbO2 cm-1/M', 'HbO2', HbO2Path) # Find HbO2 for each pixel
convolve_spectra(data, [nmStart,nmEnd], 'Hb cm-1/M', 'Hb', HbO2Path) # Find Hb for each pixel
WaterPath = '/Users/maycaj/Documents/HSI_III/Absorbances/Water Absorbance.csv'
convolve_spectra(data, [nmStart,nmEnd], 'H2O 1/cm','H2O', WaterPath)
pheoPath = '/Users/maycaj/Documents/HSI_III/Absorbances/Pheomelanin.csv'
convolve_spectra(data, [nmStart,nmEnd], 'Pheomelanin cm-1/M','Pheomelanin', pheoPath)
euPath = '/Users/maycaj/Documents/HSI_III/Absorbances/Eumelanin Absorbance.csv'
convolve_spectra(data, [nmStart,nmEnd], 'Eumelanin cm-1/M','Eumelanin', euPath)
fatPath = '/Users/maycaj/Documents/HSI_III/Absorbances/Fat Absorbance.csv'
convolve_spectra(data, [nmStart,nmEnd], 'fat', 'fat', fatPath) # Find Hb for each pixel
data['HbO2/Hb'] = data['HbO2'] / data['Hb']
col_names = col_names + ['HbO2/Hb'] # add additional column names that we will train on


def undersampleClass(data, classes, columnLabel):
    '''
    Evens out class imbalances via undersampling
    Args:
        data: pandas dataframe 
        classes: string of binary classes to undersample ['class1','class2']
        columnLabel: string label of the column where undersampling occurs
    Returns:
        undersampled data dataframe
    '''
    dataTrue = data[data[columnLabel] == classes[0]]
    dataFalse = data[data[columnLabel] == classes[1]]
    minLen = int(min(dataTrue.shape[0],dataFalse.shape[0])) # find maximum number for each class
    trueSample = dataTrue.sample(n=minLen, random_state=random_state)
    falseSample = dataFalse.sample(n=minLen, random_state=random_state)
    underSampled = pd.concat([trueSample,falseSample], axis=0)
    return underSampled

def summarizeData(data):
    '''
    prints out the sums of the weights and the sums of the samples by Id and by T/F
    args: 
        data: dataframe 
    '''
    # print out numbers of samples
    for ID in uniqIDs:
        IDsum = sum(data['ID'] == ID)
        print(f'{ID} total: {IDsum}')
    trueSum = sum(data[y_col]==y_categories[1])
    falseSum = sum(data[y_col]==y_categories[0])
    print(f'{y_categories[0]} total: {falseSum} | {y_categories[1]} total: {trueSum}')

def process_fold(foldNum, train_index, test_index, i):
    '''
    Processes each fold of the data. Is wrapped into a function so that each fold can be parallelized
    inputs:
        foldNum: what number fold we are on
        train_index: index of the training examples
        test_index: index of the testing index
        i: what number iteration we are on

    outputs: 
        confusionFold: confusion matricies for this fold
        PredLocFold: predictions and their locations for this fold
        IDaccFold: ID accuracies for this fold
        coefFold: SVM coefficients for this fold
    '''

    trainIDs = uniqIDs[train_index]
    testIDs = uniqIDs[test_index]
    print(f'Fold {foldNum} Train IDs: {trainIDs}')
    print(f'Fold {foldNum} Test IDs: {testIDs}')
    train = dataFrac[dataFrac['ID'].isin(trainIDs)]
    test = dataFrac[dataFrac['ID'].isin(testIDs)].copy()
    train = undersampleClass(train, y_categories, y_col)

    # print sum of weights and sum of examples
    print(f'\n Summarizing Training Data:')
    summarizeData(train)
    # print(f'\n Summarize Test Data')
    # summarizeData(test)

    # select the X and the y from the data
    Xtrain = train.loc[:,col_names] 
    yTrain = train[y_col]
    Xtest = test.loc[:,col_names]
    yTest = test[y_col]

    # create SVM
    if stochastic:
        svc = SGDClassifier(loss='hinge', random_state=random_state) # stocastic
    else:
        svc = SVC(kernel='linear')

    # fit SVM on data
    svc.fit(Xtrain,yTrain)
    XtestTransformed = Xtest
    yPred = svc.predict(XtestTransformed)
    coefFold = pd.DataFrame(svc.coef_[0].reshape(1,-1), columns=col_names)

    # add yTest and and yPred as columns of test
    test.loc[:,'correct'] = yTest == yPred # add a column that displays if answer is correct
    test.loc[:,'yPred'] = yPred
    test.loc[:,'yTest'] = yTest

    # Place confusion matrix info for this iteration into confusion - place all iterations in confusions
    confusionFold = pd.DataFrame([]) # find confusion matricies for this fold
    uniqTestIDs = test['ID'].unique()
    for ID in uniqTestIDs:
        # find data for this id and this foldername
        testID = test[(test['ID']==ID)] 
        if not testID.empty: # double check if this ID is present in testing
            cm = confusion_matrix(testID['yTest'],testID['yPred'], labels=y_categories)
            if cm.size != 1: # check to make sure the testing set doesn't only have one foldername
                TP, FN, FP, TN = cm.ravel()
                confusionFold = pd.concat([confusionFold, pd.DataFrame([[foldNum, ID, TN, FP, FN, TP]], columns=['foldNum','ID','TN','FP','FN','TP'])])

    # Create dataframe with ID, foldername, prediction, and coordinates for this fold
    PredLocFold = test[['X','Y','ID','yPred','FloatName','correct',y_col]].copy()
    PredLocFold['foldNum'] = foldNum

    print(f'Patch Acc Fold {foldNum} iteration {i}: {np.mean(test["correct"])}')

    # Find accuracies on this fold
    IDaccFold = test.groupby(['ID',y_col])['correct'].mean() # find average accuracy per ID
    IDaccFold = IDaccFold.to_frame(name=f'Iter {i} Fold {foldNum}') # convert to DataFrame
    IDaccFold.reset_index(inplace=True) # make 'ID' and 'Foldername' into separate columns

    return confusionFold, PredLocFold, IDaccFold, coefFold

## FIT MODELS AND SAVE OUTPUT
TN, FP, FN, TP = 0, 0, 0, 0 # initialize confusion matrix tally
uniqIDs = data['ID'].unique() # find unique IDs

# Initialize a dataframe for Location of each classification
PredLocs = pd.DataFrame([])
coefs = pd.DataFrame([])

# Initialize a dataframe for confusion matricies
confusions = pd.DataFrame(columns=['ID','TN','FP','FN','TP'])

selectedNums = [int(frac*data.shape[0]) for frac in fracs] # convert the fraction to a number
# Loop over the selected number of data and each iteration
for selectedNum in selectedNums:
    for i in range(iterations): # iterate multiple times for error bars
        print(f'Starting iteration: {i}')
        random_state = np.random.randint(0,4294967295) # generate random integer for random state
        dataFrac = data.sample(n=selectedNum,random_state=random_state) # include a fraction of the data 
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state) #shuffle is important to avoid bias
        
        # Run each fold in parallel
        foldOutput = Parallel(n_jobs=n_jobs)(
            delayed(process_fold)(foldNum, train_index, test_index, i)
            for foldNum, (train_index, test_index) in enumerate(kf.split(uniqIDs))
        )
        # unpack then process output
        confusionFold, PredLocFold, IDaccFold, coefFold = zip(*foldOutput) 
        confusionFold = pd.concat(confusionFold, ignore_index=True)
        PredLocFold = pd.concat(PredLocFold, ignore_index=True)
        IDaccFold = pd.concat(IDaccFold, ignore_index=True)
        coefFold = pd.concat(coefFold, ignore_index=True)

        # add outputs to a dataframe for all of the predictions combined
        confusions = pd.concat([confusions, confusionFold])
        PredLocs = pd.concat([PredLocs, PredLocFold])
        # coefFold.columns = [col.split('_')[1] for col in coefFold.columns] # Change all of the columns to be just numbers
        coefs = pd.concat([coefs,coefFold])
        if i == 0:
            IDaccs = IDaccFold
        else: 
            IDaccs = IDaccs.merge(IDaccFold, on=['ID',y_col], how='left')
        pass
print('Done with Training')

## PROCESS OUTPUT DATA
def roundCells(cell, thresh):
    '''
    rounds cells to desired threshold
    args:
        cell: cell of a dataframe
        thresh: threshold decimal
    returns:
        output: rounded cells
    '''
    output =  np.where(np.isnan(cell), np.nan, np.where(cell > thresh, 1, 0)) # np.round() is used later on so it must be < not <=
    return output

#Add averages to dataframe
IDavg = IDaccs.drop(['ID',y_col], axis=1).T.mean() #Exclude the non-numerical data, and find mean across IDs using transpose
IDaccs.insert(2,'ID Avg',IDavg) # Insert ID average
for thresh in [threshold]: # np.arange(0.3,0.7,0.1): # iterate over possible thresholds
    thresh = np.round(thresh, 1)
    IDroundAvg = IDaccs.drop(['ID',y_col], axis=1).T.apply(lambda cell: roundCells(cell,thresh)).mean() # round cells to threshold (thresh) and then take the mean across the rows
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
IDaccs.loc['Column Average'] = IDaccs.drop(['ID',y_col], axis=1).mean(axis=0) 

# apply 95% CI to iterations data
hasFolderHasAvg = (IDaccs.loc[:,y_col].notna()) | (IDaccs.index == 'Column Average') # select rows with folder names and the column average
iteration = IDaccs.loc[hasFolderHasAvg, 'Iter 0 Fold 0':f'Iter {i} Fold {n_splits-1}'] # select all of iteration data
# iteration = IDaccs.loc[hasFolderHasAvg, 'Iter 0 Fold 0':f'Iter {i} Fold {foldNum}'] # select all of iteration data
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
IDaccs.insert(2,'Labels','ID: ' + IDaccs['ID'].astype(str) + '\n' + 'Cat: ' + IDaccs[y_col].astype(str)) # combine columns for figure labels
IDaccs.loc['Column Average','Labels'] = 'Column Average'

# Find accuracy by Foldername; Add to IDaccs
FolderAcc = IDaccs.groupby(y_col)['ID Avg'].mean() # find accuracy by foldername
FolderAcc = FolderAcc.reset_index(inplace=False)
FolderAcc['Labels'] = FolderAcc[y_col]
IDaccs = pd.concat([IDaccs, FolderAcc], axis=0)

# Find rounded accuracy by Foldername; Add to IDaccs
FolderRound = IDaccs.groupby(y_col)['ID 0.5 Rounded Avg'].mean() # rounded accuracy grouped by y_col categories
FolderRound = FolderRound.reset_index(inplace=False)
IDaccs.loc[IDaccs['Labels']==y_categories[1],'ID 0.5 Rounded Avg'] = FolderRound.loc[FolderRound[y_col]==y_categories[1],'ID 0.5 Rounded Avg']
IDaccs.loc[IDaccs['Labels']==y_categories[0],'ID 0.5 Rounded Avg'] = FolderRound.loc[FolderRound[y_col]==y_categories[0],'ID 0.5 Rounded Avg']

end_time = time.time()
print(f'Total time: {end_time-start_time}')


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
    labels = labels.astype(str)
    values = df[hasValues][valuesKey] # y values for bar chart
    yerrPlot = df[hasValues][errorKey]
    yerrPlot = np.vstack(yerrPlot).T # errors for bar chart
    # plot bar chart
    fig, ax = plt.subplots()
    bars = ax.bar(labels, values, yerr=yerrPlot)
    ax.tick_params(axis='x', labelsize=5, rotation=90)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title + ' overall average=' + str(round(values['Column Average']*100,1))) 
    plt.tight_layout()
    return fig

# wholeLeg = columnBarPlot(IDaccs, IDaccs['ID 0.5 Rounded Avg'].notna(),'Labels','ID 0.5 Rounded Avg','YerrRounded',f'ID \n {y_col}','Accuracy',f'{y_col} Leg Acc n={selectedNum} iterations={iterations}')
patch = columnBarPlot(IDaccs, IDaccs['ID Avg'].notna(),'Labels','ID Avg','Yerr',f'ID \n {y_col}','Accuracy',f'{y_col} Patch Acc n={selectedNum} iterations={iterations} fracs={fracs} \n {filepath.split("/")[-1]}')

## Plot the SVM coefficients
plt.figure() 
plt.plot(abs(coefs).median(axis=0))
plt.xticks(rotation=90)
plt.title(f'Testing out HbO2 and Hb as features with {filepath.split("/")[-1]} \n iterations: {iterations} n={selectedNum} fracs={fracs}')
plt.tight_layout()
plt.show()

# Find the confusion matricies for each ID
confusion = confusions.groupby('ID').sum()
confusion = confusion.reset_index()

# add metadata and save 
date = filepath.split('/')[-1].split('_')[0] # find date from file name
IDaccs.loc['Info','Labels'] = f'input filename: {filepath.split("/")[-1]}' # add input filename

# save all of CSVs and pdfs
IDaccs.to_csv(f'/Users/maycaj/Downloads/{date}IDaccs_n={selectedNum}i={iterations}.csv') # Save the accuracy as csv
PredLocs.to_csv(f'/Users/maycaj/Downloads/{date}PredLocs_n={selectedNum}i={iterations}.csv.gz', compression='gzip', index=False) # Save predictions with locations as csv
confusion.to_csv(f'/Users/maycaj/Downloads/{date}confusion_n={selectedNum}i={iterations}.csv')
confusions.to_csv(f'/Users/maycaj/Downloads/{date}confusions_n={selectedNum}i={iterations}.csv')
coefs.to_csv(f'/Users/maycaj/Downloads/{date}coefs_n={selectedNum}i={iterations}.csv', index=False)
# wholeLeg.savefig(f'/Users/maycaj/Downloads/{date}wholeLeg_n={selectedNum}i={iterations}.pdf')
patch.savefig(f'/Users/maycaj/Downloads/{date}patch_n={selectedNum}i={iterations}.pdf')
print('All done ;)')