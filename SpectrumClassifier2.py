#!/Users/maycaj/Documents/HSI_III/.venv/bin/python3

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
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
import time
from joblib import Parallel, delayed
import matplotlib.colors as mcolors
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error




## INITIALIZE PARAMETERS
bioWulf = False # If we are running on bioWulf supercomputer
if bioWulf:
    filepath = '/data/maycaj/PatchCSVs/Feb21_1x1_ContinuumRemoved_WLFiltered_Recrop_PAREINC.csv'
else:
    filepath = '/Users/maycaj/Documents/HSI_III/PatchCSVs/Feb4_1x1_ContinuumRemoved_WLFiltered.csv' #'/Users/maycaj/Documents/HSI_III/Mar4_1x1_ContinuumRemoved_WLFiltered_Recrop_Gender.csv' #'/Users/maycaj/Documents/HSI_III/Mar4_1x1_ContinuumRemoved_WLFiltered_Recrop_Gender.csv' # '/Users/maycaj/Documents/HSI_III/Mar10_Noarm.csv' # '/Users/maycaj/Documents/HSI_III/Feb21_1x1_ContinuumRemoved_WLFiltered_Recrop_PAREINC.csv' #'/Users/maycaj/Documents/HSI_III/1-22-25_5x5.csv'
fracs = [0.01] # fraction of patches to include
iterations = 10 # how many times to repeat the analysis
threshold = 0.5 # what fraction of patches of edemaTrue to consider whole leg edemaTrue
nmStartEnd = ['Wavelength_451.18','Wavelength_954.83'] # ['Wavelength_760.61','Wavelength_760.61']# ['Wavelength_451.18','Wavelength_954.83'] # specify the wavelengths to include in analysis (1x1 continuum)
# nmStartEnd = ['451.18','954.83'] 
n_jobs = -1 # Number of CPUs to use during each Fold. -1 = as many as possible
n_splits = 10 # number of splits to make the fold
n_components = 40 # Number of PCA components
stochastic = False
y_col = 'Foldername' # 'patient_height' #'patient_skin_type' # 'Gender'

start_time = time.time()
print('Loading dataframe...')
data = pd.read_csv(filepath)
print(f'Done loading dataframe ({np.round(time.time()-start_time,1)}s)')
y_categories = data[y_col].unique() # ['Male','Female'] # [0,1] is [short, tall] and [pale, dark]

## PROCESS DATA FOR CERTAIN y
# data = data[data['final_diagnosis_other'] == 'Cellulitis'] # select the cellulitis condition only

# data['patient_skin_type'] = data['patient_skin_type'].apply(lambda x: 0 if x <= 2 else 1) # divide patients evenly by skin type
data = data[(data['ID'].isin([11,12,15,18,20,22,23,26,34,36])) | (data['Foldername'] =='EdemaFalse') ] # Select only cellulitis IDs or IDs in EdemaFalse Folder
# data = data[(data['ID'].isin([1,2,6,7,9,10,13,14,19,24,27,29,30,31,32,33,35,37,38,39,40])) | (data['Foldername'] =='EdemaFalse') ] # Select only Peripheral IDs or IDs in EdemaFalse Folder
# data = pd.concat([data.loc[:,nmStartEnd[0]:nmStartEnd[1]],data['ID'], data['Foldername']], axis=1)
# data = data.groupby(['ID','Foldername'], as_index=False).median()

# keep this section together for height: need to remove Dr. pare and the other patients who don't have heights
# data = data[data['ID'] != 0] # Remove Dr. Pare
# data = data.dropna(subset=['patient_height']); # Remove all patients that do not have heights
# data['patient_height'] = data['patient_height'].apply(lambda x: 0 if x < 68 else 1) # 1 = tall (above 68in)

## INITIALIZE FUNCTIONS
def convolve_spectra(data, nmStartEnd, spectraName, outputName, absorbPath, showFig):
    '''
    Takes dataframe and convolves the spectra with Hemoglobin's response curve to see the total absorbance of hemoglobin in the skin.
    Args:
        data: main pandas dataframe with wavelength and demographic data
        nmStartEnd: [startingWavelength, endingWavelength] in data
        spectraName: Name of column in the absorbance csv
        absorbPath: path to .csv with the absorbances
        showFig: boolean whether or not to plot
    '''
    absorbCSV = pd.read_csv(absorbPath) # Load the HbO2 spectra
    skinWaves = list(data.loc[:,nmStartEnd[0]:nmStartEnd[1]].columns)
    skinWaves = [float(wavelength.split('_')[1]) for wavelength in skinWaves] # Find wavelengths in skindata

    absorbWaves = []
    for wavelength in skinWaves: # match the data wavelengths with the wavelengths on the absorbance spectra
        absorbWaves.append(min(absorbCSV['lambda nm'], key=lambda x: np.abs(x-wavelength))) 
    matchedWave = pd.DataFrame({'AbsorbWaves': absorbWaves, 'SkinWaves': skinWaves})

    # remove the Skin Wave Values that are outside of the range of the absorbance values
    maxSkin = float('inf')
    minSkin = float('inf')
    for wavelength in skinWaves: # find the maximum and minimum wavelengths in the skin waves
        # Looking for the closest absorbance to the highest skin wavelength
        maxDiff = np.abs(max(matchedWave['AbsorbWaves'])-wavelength) # absolute difference between highest absorbance wavelength and each skin wavelength
        if maxDiff < maxSkin: # if the absolute difference is lower than all the others we have seen before, update the threshold for the absolute difference, and find the new max wavelength
            maxSkin = maxDiff
            maxWavelength = wavelength
        # Looking for the closest absorbance to the lowest skin wavelength
        minDiff = np.abs(min(matchedWave['AbsorbWaves'])-wavelength)
        if minDiff < minSkin:
            minSkin = minDiff
            minWavelength = wavelength

    matchedWave = matchedWave[matchedWave['SkinWaves'] >= minWavelength]
    matchedWave = matchedWave[matchedWave['SkinWaves'] <= maxWavelength]

    matchedWave = matchedWave.merge(absorbCSV, how='left', left_on='AbsorbWaves', right_on='lambda nm')

    matchedAbsorbs = matchedWave[spectraName].values
    # matchedAbsorbs = matchedAbsorbs/np.max(matchedAbsorbs) # Normalize absorbance values from 0 to 1
    data_selected = data.loc[:,'Wavelength_' + f"{minWavelength:.2f}":'Wavelength_' + f"{maxWavelength:.2f}"]
    data_selected = data_selected.values
    data[outputName] = np.dot(data_selected, matchedAbsorbs)

    # # Normalize output
    # data[outputName] = data[outputName] - data[outputName].min()
    # data[outputName] = data[outputName] / data[outputName].max()

    if showFig: # plot the absorbance spectra next 
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=skinWaves, y=matchedAbsorbs, name=outputName, line=dict(color='green'))) # plot HbO2
        for i in range(data.shape[0]): # plot all of skin spectra
            ID = str(data.iloc[i,data.columns.get_loc('ID')])
            HbO2 = str(np.round(data.iloc[i, data.columns.get_loc(outputName)],2))
            Foldername = str(data.iloc[i,data.columns.get_loc('Foldername')])
            name = 'ID:' + ID + outputName + HbO2 + 'Foldername:' + Foldername
            if Foldername == 'EdemaTrue':
                line=dict(color='red')
            elif Foldername == 'EdemaFalse':
                line=dict(color='blue')
            fig.add_trace(go.Scatter(x=skinWaves, y=data.iloc[i,data.columns.get_loc(nmStartEnd[0]):data.columns.get_loc(nmStartEnd[1])+1], name=name, line=line))
        fig.update_layout(title='Skin Spectra with HbO2 convolution as label and HbO2 spectra')
        fig.show()
    return matchedAbsorbs

HbO2Path = '/Users/maycaj/Documents/HSI_III/Absorbances/HbO2 Absorbance.csv'
convolve_spectra(data, [nmStartEnd[0],nmStartEnd[-1]], 'HbO2 cm-1/M', 'HbO2', HbO2Path, showFig=False) # Find HbO2 for each pixel
convolve_spectra(data, [nmStartEnd[0],nmStartEnd[-1]], 'Hb cm-1/M', 'Hb', HbO2Path, showFig=False) # Find Hb for each pixel


data['Hb/HbO2'] = data['Hb'] / data['HbO2'] # Find hemoglobin oxygen saturation ratio from the ratio of oxyhemoblobin to total hemoglobin: https://onlinelibrary.wiley.com/doi/full/10.1111/srt.12074


def plotSpectra(data, groupBy, plotID=False):
    '''
    Plots the average spectra by a certain group and also includes the average spectra for each ID
    Args:
        groupBy: the column of data to group by
        data: input dataframe
        plotID: plot each ID on the legend
    '''
    # isolate wavelengths we want to plot and group together via the groupBy variable
    goodWavelengths = data.loc[:,nmStartEnd[0]:nmStartEnd[1]] # remove noisy wavelengths
    data2plot = pd.concat([data[groupBy], goodWavelengths], axis=1) 
    groupedData = data2plot.groupby(groupBy).mean()
    groupedData.reset_index(inplace=True)

    # initialize color space
    n_groups = len(groupedData[groupBy].unique())
    # Create a truncated Viridis colormap (removing the top 20%)
    viridis_truncated = mcolors.LinearSegmentedColormap.from_list(
        "viridis_truncated", plt.cm.viridis(np.linspace(0, 0.8, 256))
    )
    # Generate colors from the truncated colormap
    colors = viridis_truncated(np.linspace(0, 1, n_groups))
    # colors = plt.cm.viridis(np.linspace(0, 1, n_groups))
    plt.figure()
    for i, color in zip(range(groupedData.shape[0]),colors): # iterate over each example in groupedData and its corresponding color
        # plot the mean of each category
        y = groupedData.loc[:,nmStartEnd[0]:nmStartEnd[1]]
        cur_category = groupedData[groupBy].iloc[i]
        plt.plot(list(goodWavelengths.columns), y.iloc[i,:], label=cur_category, color=color)

        # plot each of the ID's spectra for a certain within the categroy
        if (groupBy != 'ID') and plotID: 
            data2plot2 = pd.concat([data[groupBy], data['ID'], goodWavelengths], axis=1)
            IDspectra = data2plot2.groupby(['ID',groupBy]).mean()
            IDspectra.reset_index(inplace=True)
            IDspectra = IDspectra[IDspectra[groupBy] == cur_category] # select this group
            for j in range(IDspectra.shape[0]):
                plt.plot(list(goodWavelengths.columns), IDspectra.iloc[j,:].drop(['ID',groupBy]), color=color, alpha=0.1)
            pass
    plt.title(f'Mean spectra grouped for each ID by {groupBy}')
    plt.xticks(rotation=90)
    plt.legend()
    pass
# plotSpectra(data, 'ID')
# plotSpectra(data, 'Gender', True)
# plotSpectra(data, 'patient_height', True)
# plotSpectra(data, 'patient_skin_type', True)
# plotSpectra(data, 'Foldername', True)
# plt.show()


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

def splitByLeg(data):
    '''
    splits each ID so each condition (EdemaTrue vs EdemaFalse) is randomly separated between training and testing
    inputs:
        data: hyperspectral dataframe
    outputs:
        trainData: training set
        testData: testing set
    '''
    uniqIDs = data['ID'].unique()
    trainData = pd.DataFrame([])
    testData = pd.DataFrame([])
    for ID in uniqIDs:
        if np.random.rand() > 0.5: # add EdemaTrue to train and EdemaFalse to test
            trainData = pd.concat([data[(data['ID'] == ID) & (data['Foldername'] == 'EdemaTrue')],trainData])
            testData = pd.concat([data[(data['ID'] == ID) & (data['Foldername'] == 'EdemaFalse')],testData])
        else: # add EdemaTrue to test and EdemaFalse to train
            testData = pd.concat([data[(data['ID'] == ID) & (data['Foldername'] == 'EdemaTrue')],testData])
            trainData = pd.concat([data[(data['ID'] == ID) & (data['Foldername'] == 'EdemaFalse')],trainData])
    return trainData, testData


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


def summarizeData(data):
    '''
    prints out the sums of the weights and the sums of the samples by Id and by T/F
    args: 
        data: dataframe 
    '''
    # # Print out the sums of the weights for: 1) Each ID 2) T vs F
    # for ID in uniqIDs:
    #     IDweight = sum(data.loc[data['ID'] == ID,'Weights'])
    #     print(f'{ID} weight sum: {IDweight}')

    # trueWeight =  sum(data.loc[data['Foldername']=='EdemaTrue','Weights'])
    # falseWeight = sum(data.loc[data['Foldername']=='EdemaFalse','Weights'])
    # print(f'True Weight sum: {trueWeight}')
    # print(f'False Weight sum: {falseWeight}')

    # print out numbers of samples
    for ID in uniqIDs:
        IDsum = sum(data['ID'] == ID)
        print(f'{ID} total: {IDsum}')
    trueSum = sum(data[y_col]==y_categories[1])
    falseSum = sum(data[y_col]==y_categories[0])
    print(f'{y_categories[0]} total: {falseSum} | {y_categories[1]} total: {trueSum}')

def plotScree(pca, title):
    '''
    Plot how much variance is explained by PCA
    args: 
        pca: PCA pipeline object from sklearn
    '''
    print(pca.named_steps['pca'].explained_variance_ratio_)
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(pca.named_steps['pca'].explained_variance_ratio_) + 1), pca.named_steps['pca'].explained_variance_ratio_)
    plt.xlabel('Number of Components')
    plt.ylabel('Explained Variance Ratio')
    plt.title(title)

def plotPCAcomponents(pca):
    '''    
    plot the PCA weights
    args: 
        pca: PCA pipeline object from sklearn
    '''
    components = pca.named_steps['pca'].components_
    components = np.abs(components)
    wavelengths = [451.18,456.19,461.21,466.23,471.25,476.28,481.32,486.36,491.41,496.46,501.51,506.57,511.64,516.71,521.78,526.86,531.95,537.04,542.13,547.23,552.34,557.45,562.57,567.69,572.81,577.94,583.07,588.21,593.36,598.51,603.66,608.82,613.99,619.16,624.33,629.51,634.7,639.88,645.08,650.28,655.48,660.69,665.91,671.12,676.35,681.58,686.81,692.05,697.29,702.54,707.8,713.06,718.32,723.59,728.86,734.14,739.42,744.71,750.01,755.3,760.61,765.92,771.23,776.55,781.87,787.2,792.53,797.87,803.21,808.56,813.91,819.27,824.63,830,835.37,840.75,846.13,851.52,856.91,862.31,867.71,873.12,878.53,883.95,889.37,894.8,900.23,905.67,911.11,916.56,922.01,927.47,932.93,938.4,943.87,949.35,954.83]
    for j in range(components.shape[0]):
        plt.scatter(wavelengths,components[j], label=f'Component: {j}')
        if j > 2: # don't plot more than 3 dimensions
          break   
    plt.legend()
    plt.title('PCA weights')
    plt.xlabel('Band (nm)')
    plt.ylabel('Absolute value of Weight')



## FIT MODELS AND SAVE OUTPUT
TN, FP, FN, TP = 0, 0, 0, 0 # initialize confusion matrix tally
uniqIDs = data['ID'].unique() # find unique IDs

# Initialize a dataframe for Location of each classification
PredLocs = pd.DataFrame([])
coefs = pd.DataFrame([])

# Initialize a dataframe for confusion matricies
confusions = pd.DataFrame(columns=['ID','TN','FP','FN','TP'])

selectedNums = [int(frac*data.shape[0]) for frac in fracs] # convert the fraction to a nu
# Loop over the selected number of data and each iteration
for selectedNum in selectedNums:
    for i in range(iterations): # iterate multiple times for error bars
        print(f'Starting iteration: {i}')
        random_state = np.random.randint(0,4294967295) # generate random integer for random state

        dataFrac = data.sample(n=selectedNum,random_state=random_state) # include a fraction of the data so debugging is fast

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state) #shuffle is important to avoid bias

        col_names = list(data.loc[:,nmStartEnd[0]:nmStartEnd[1]].columns) # ['Hb/HbO2', 'Hb','HbO2' ] + list(data.loc[:,nmStartEnd[0]:nmStartEnd[1]].columns) # find names of all of the columns that we will use for training.

        def process_fold(foldNum, train_index, test_index, i):
            trainIDs = uniqIDs[train_index]
            testIDs = uniqIDs[test_index]
            print(f'Fold {foldNum} Train IDs: {trainIDs}')
            print(f'Fold {foldNum} Test IDs: {testIDs}')
            train = dataFrac[dataFrac['ID'].isin(trainIDs)]
            test = dataFrac[dataFrac['ID'].isin(testIDs)].copy()
            
            # # within each ID, divide by condition such that EdemaTrue is in train and EdemaFalse is in test or vice versa
            # train, _ = splitByLeg(train)

            # Undersample the majority class so there are equal numbers of each class
            # train = undersampleClass(train, ['EdemaTrue','EdemaFalse'], 'Foldername')
            train = undersampleClass(train, y_categories, y_col)

            # print sum of weights and sum of examples
            summarizeData(train)

            # select the X and the y from the data
            Xtrain = train.loc[:,col_names] 
            # yTrain = train['Foldername']
            yTrain = train[y_col]
            Xtest = test.loc[:,col_names]
            # yTest = test['Foldername']
            yTest = test[y_col]

            # # do PCA dimensionality reduction
            # pca = make_pipeline(PCA(n_components=n_components, random_state=random_state))

            # pca.fit(Xtrain, yTrain) # fit method's model

            # # inspect PCA
            # plotScree(pca, f'Scree Plot - Examples per iteration: {selectedNum}')
            # plotPCAcomponents(pca)
            # plt.show()

            # create SVM
            if stochastic:
                svc = SGDClassifier(loss='hinge', random_state=random_state) # stocastic
            else:
                svc = SVC(kernel='linear')  

            # fit SVM on data
            # svc.fit(pca.transform(Xtrain),yTrain)
            # XtestTransformed = pca.transform(Xtest)
            svc.fit(Xtrain,yTrain)
            XtestTransformed = Xtest
            yPred = svc.predict(XtestTransformed)
            coef = pd.DataFrame(svc.coef_[0].reshape(1,-1), columns=col_names)
            print(svc.classes_)
            pass

            # ## Partial Least Squares
            # # Binarize yTrain and yTest 
            # yTrain = np.where(yTrain == y_categories[0], 1, 0) 
            # yTest = np.where(yTest==y_categories[0], 1, 0)
            # # Fit patial least squares model
            # pls_model = PLSRegression(n_components=3) 
            # pls_model.fit(Xtrain, yTrain) # ____ I think that this will throw an error because yTrain is not binarized 
            # yPred = pls_model.predict(Xtest)
            # # Evaluate the model performance
            # r_squared = pls_model.score(Xtest, yTest)
            # print(f'R-Squared Error: {r_squared}')
            # mse = mean_squared_error(yTest, yPred)
            # print(f'Mean Squared Error: {mse}')
            # plt.scatter(yTest, yPred, alpha=0.002)
            # plt.xlabel('Actual (0=EdemaFalse, 1=EdemaTrue)')
            # yPred = np.array(yPred.round()) # Round the prediction such that the actual and predicted can be compared

            # add yTest and and yPred as columns of test
            test.loc[:,'correct'] = yTest == yPred # add a column that displays if answer is correct
            plt.title(f'Partial Least Squares Regression: Cellulitis \n Accuracy={test["correct"].mean().round(2)*100}%')
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
                    # cm = confusion_matrix(testID['yTest'],testID['yPred'], labels=['EdemaTrue','EdemaFalse'])
                    if cm.size != 1: # check to make sure the testing set doesn't only have one foldername
                        TN, FP, FN, TP = cm.ravel()
                        # confusions = pd.concat([confusions, pd.DataFrame([[ID, foldername, TN, FP, FN, TP]], columns=['ID','Foldername','TN','FP','FN','TP'])])
                        confusionFold = pd.concat([confusionFold, pd.DataFrame([[foldNum, ID, TN, FP, FN, TP]], columns=['foldNum','ID','TN','FP','FN','TP'])])
                    pass

            # Create dataframe with ID, foldername, prediction, and coordinates for this fold
            PredLocFold = test[['X','Y','ID','yPred','FloatName','correct',y_col]].copy()
            PredLocFold['foldNum'] = foldNum

            print(f'Patch Acc Fold {foldNum} iteration {i}: {np.mean(test["correct"])}')

            # Find accuracies on this fold
            IDaccFold = test.groupby(['ID',y_col])['correct'].mean() # find average accuracy per ID
            IDaccFold = IDaccFold.to_frame(name=f'Iter {i} Fold {foldNum}') # convert to DataFrame
            IDaccFold.reset_index(inplace=True) # make 'ID' and 'Foldername' into separate columns

            return confusionFold, PredLocFold, IDaccFold, coef
        
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

        # add outputs to 
        confusions = pd.concat([confusions, confusionFold])
        PredLocs = pd.concat([PredLocs, PredLocFold])
        coefFold.columns = [col.split('_')[1] for col in coefFold.columns] # Change all of the columns to be just numbers
        coefs = pd.concat([coefs,coefFold])
        if i == 0:
            IDaccs = IDaccFold
        else: 
            IDaccs = IDaccs.merge(IDaccFold, on=['ID',y_col], how='left')
        pass


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
    return fig

wholeLeg = columnBarPlot(IDaccs, IDaccs['ID 0.5 Rounded Avg'].notna(),'Labels','ID 0.5 Rounded Avg','YerrRounded',f'ID \n {y_col}','Accuracy',f'{y_col} Leg Acc n={selectedNum} iterations={iterations}')
patch = columnBarPlot(IDaccs, IDaccs['ID Avg'].notna(),'Labels','ID Avg','Yerr',f'ID \n {y_col}','Accuracy',f'{y_col} Patch Acc n={selectedNum} iterations={iterations}')
plt.show()

# Find the confusion matricies for each ID
confusion = confusions.groupby('ID').sum()
confusion = confusion.reset_index()

# add metadata and save 
date = filepath.split('/')[-1].split('_')[0] # find date from file name
IDaccs.loc['Info','Labels'] = f'input filename: {filepath.split("/")[-1]}' # add input filename

# save all of CSVs and pdfs
if bioWulf:
    IDaccs.to_csv(f'/data/maycaj/output/{date}IDaccs_n={selectedNum}i={iterations}.csv') # Save the accuracy as csv
    PredLocs.to_csv(f'/data/maycaj/output/{date}PredLocs_n={selectedNum}i={iterations}.csv.gz', compression='gzip', index=False) # Save predictions with locations as csv
    confusion.to_csv(f'/data/maycaj/output/{date}confusion_n={selectedNum}i={iterations}.csv')
    confusions.to_csv(f'/data/maycaj/output/{date}confusions_n={selectedNum}i={iterations}.csv')
    wholeLeg.savefig(f'/data/maycaj/output/{date}wholeLeg_n={selectedNum}i={iterations}.pdf')
    patch.savefig(f'/data/maycaj/output/{date}patch_n={selectedNum}i={iterations}.pdf')
else: # on local computer
    IDaccs.to_csv(f'/Users/maycaj/Downloads/{date}IDaccs_n={selectedNum}i={iterations}.csv') # Save the accuracy as csv
    PredLocs.to_csv(f'/Users/maycaj/Downloads/{date}PredLocs_n={selectedNum}i={iterations}.csv.gz', compression='gzip', index=False) # Save predictions with locations as csv
    confusion.to_csv(f'/Users/maycaj/Downloads/{date}confusion_n={selectedNum}i={iterations}.csv')
    confusions.to_csv(f'/Users/maycaj/Downloads/{date}confusions_n={selectedNum}i={iterations}.csv')
    coefs.to_csv(f'/Users/maycaj/Downloads/{date}coefs_n={selectedNum}i={iterations}.csv', index=False)
    wholeLeg.savefig(f'/Users/maycaj/Downloads/{date}wholeLeg_n={selectedNum}i={iterations}.pdf')
    patch.savefig(f'/Users/maycaj/Downloads/{date}patch_n={selectedNum}i={iterations}.pdf')
print('All done ;)')