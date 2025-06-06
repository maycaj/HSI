#!/Users/maycaj/Documents/HSI_III/.venv/bin/python3

### Classify each individual pixel from the hyperspectral image using an SVM. The input data is in CSV form. Ouputs are bootstrapped accuracy bar charts and SVM coefficients.

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold
import time
from joblib import Parallel, delayed
import pyarrow.csv as pv
from ConvolveSpectra import chrom_dot_spectra
from datetime import datetime
import sys
import os
from pathlib import Path
import shutil
# from LeastSquares import getLeastSquares

class training:
    @staticmethod
    def loadDF(filepath):
        '''
        Args: 
            filepath (str): path to df
        Returns:
            df (DataFrame)
        '''
        ## Load DataFrame
        print('Loading dataframe...')
        table = pv.read_csv(filepath)
        df = table.to_pandas()
        print(f'Done loading dataframe ({np.round(time.time()-start_time,1)}s)')
        return df 

    @staticmethod
    def undersampleClass(df, classes, columnLabel):
        '''
        Evens out class imbalances via undersampling
        Args:
            df: pandas dataframe 
            classes: string of binary classes to undersample ['class1','class2']
            columnLabel: string label of the column where undersampling occurs
        Returns:
            undersampled data dataframe
        '''
        dfTrue = df[df[columnLabel] == classes[0]]
        dfFalse = df[df[columnLabel] == classes[1]]
        minLen = int(min(dfTrue.shape[0],dfFalse.shape[0])) # find maximum number for each class
        trueSample = dfTrue.sample(n=minLen, random_state=random_state)
        falseSample = dfFalse.sample(n=minLen, random_state=random_state)
        underSampled = pd.concat([trueSample,falseSample], axis=0)
        return underSampled

    @staticmethod
    def summarizeData(df):
        '''
        prints out the sums of the weights and the sums of the samples by Id and by T/F
        args: 
            df: dataframe 
        '''
        # print out numbers of samples
        for ID in uniqIDs:
            IDsum = sum(df['ID'] == ID)
            print(f'{ID} total: {IDsum}')
        trueSum = sum(df[y_col]==y_categories[1])
        falseSum = sum(df[y_col]==y_categories[0])
        print(f'{y_categories[0]} total: {falseSum} | {y_categories[1]} total: {trueSum}')

    @staticmethod
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
        train = training.undersampleClass(train, y_categories, y_col) # Why is undersampling occuring on only edemaFalse?

        # print sum of weights and sum of examples
        print(f'\n Summarizing Training Data:')
        training.summarizeData(train)
        # print(f'\n Summarize Test Data')
        # summarizeData(test)

        # select the X and the y from the data
        Xtrain = train.loc[:,col_names] 
        yTrain = train[y_col]
        Xtest = test.loc[:,col_names]
        yTest = test[y_col]

        svc = SVC(kernel='linear') 
        # svc = SVC(kernel='rbf')

        # fit SVM on data
        if scale:
            scaler = StandardScaler()
            Xtrain = scaler.fit_transform(Xtrain)
        svc.fit(Xtrain,yTrain)
        if scale:
            Xtest = scaler.transform(Xtest) # Scale test data using the same scaler
        yPred = svc.predict(Xtest)

        # svc.fit(Xtrain, yTrain)
        # yPred = svc.predict(Xtest)

        coefFold = pd.DataFrame(svc.coef_[0].reshape(1,-1), columns=col_names)
        # coefFold = pd.DataFrame(np.zeros((1,128)))

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
        PredLocFold = test[['ID','yPred','FloatName','correct',y_col]].copy()
        PredLocFold['foldNum'] = foldNum

        print(f'Patch Acc Fold {foldNum} iteration {i}: {np.mean(test["correct"])}')

        # Find accuracies on this fold
        IDaccFold = test.groupby(['ID',y_col])['correct'].mean() # find average accuracy per ID
        IDaccFold = IDaccFold.to_frame(name=f'Iter {i} Fold {foldNum}') # convert to DataFrame
        IDaccFold.reset_index(inplace=True) # make 'ID' and 'Foldername' into separate columns

        return confusionFold, PredLocFold, IDaccFold, coefFold

class postprocessing:
    ## PROCESS OUTPUT DATA
    @staticmethod
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

    @staticmethod
    def bootstrapSeries(series, round=False):
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
        series = series[series.notna()].values # remove nan values
        if round:
            series = np.round(series)
        for _ in range(100): #___ set to 1000 for deployment
            sample = np.random.choice(series, size=len(series), replace=True)
            bootstrapped_means.append(np.mean(sample))
        lower = np.percentile(bootstrapped_means, (1-confidence)*100/2)
        upper = np.percentile(bootstrapped_means, (1+confidence)*100/2)
        upperLower = np.array([np.round(float(lower),2), np.round(float(upper),2)])
        return upperLower

    @staticmethod
    def halfDifference(input):
        '''
        construct yerror 
        input: [lowerCI, upperCI]
        output: half of difference between lower CI and upper CI
        '''
        return np.abs(input[0]-input[1])/2

    @staticmethod
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
        acc = str(round(values['Column Average']*100,1))
        plt.title(title + ' Average=' + acc)
        plt.tight_layout()
        return fig, acc

if __name__ == '__main__':
    ## INITIALIZE PARAMETERS
    ## fracs: fraction to keep in the analysis
    # fracs, filepath = [0.001], '/Users/maycaj/Documents/HSI/PatchCSVs/May_29_NOCR_FullRound1and2AllWLs.csv' #'/Users/maycaj/Documents/HSI_III/PatchCSVs/Mar25_NoCR_PostDeoxyCrop.csv' # '/Users/maycaj/Documents/HSI_III/PatchCSVs/Mar21_CR_PostDeoxyCrop.csv'
    fracs, filepath = [1], '/Users/maycaj/Documents/HSI/PatchCSVs/May_29_NOCR_FullRound1and2AllWLs_medians.csv' #' # for fast debugging
    iterations = 3 # how many times to repeat the analysis
    n_jobs = 1 # Number of CPUs to use during each Fold. -1 = as many as possible
    y_col = 'Foldername' 
    scale = False # To use StandardScaler() in the script
    nmStart = 451.18
    nmEnd = 1004.39 #603.66 #954.83
    start_time = time.time()
    df = training.loadDF(filepath)

    ## PROCESS DATA IDs
    print('Selecting IDs')
    dataConfig = 'Round 1 & 2: peripheral or edemafalse'
    # Store data configurations in a {'Label': (IDs...)} format
    dataConfigs = {'Round 1: cellulitis or edemafalse': (11,12,15,18,20,22,23,26,34,36), 
                'Round 1: peripheral or edemafalse': (1,2,5,6,7,8,9,10,13,14,19,21,24,27,29,30,31,32,33,35,37,38,39,40),
                'Round 1 & 2: cellulitis or edemafalse': (11,12,15,18,20,22,23,26,34,36,45,59,61,70),
                'Round 1 & 2: peripheral or edemafalse': (1,2,5,6,7,8,9,10,13,14,19,21,24,27,29,30,31,32,33,35,37,38,39,40,41,42,43,46,47,48,49,51,53,54,55,56,57,58,60,62,63,64,65,66,67,68,69,71,72)}
    df = df[(df['ID'].isin(dataConfigs[dataConfig]) | (df['Foldername'] == 'EdemaFalse'))]

    df = df[df['ID'] != 0] # Remove Dr. Pare
    n_splits = len(df['ID'].unique()) # number of splits to make the fold
    col_names = [col for col in df.columns if ((col.startswith('Wavelength_')))]
    col_names = [col for col in col_names if (float(col.split('_')[1]) >= nmStart) & (float(col.split('_')[1]) <= nmEnd) ]
    y_categories = df[y_col].unique() # ['Male','Female'] # [0,1] is [short, tall] and [pale, dark]

    ## Apply linear least squares to add new chromophore features
    # start_ls = time.time()
    # print('Starting least squares...')
    # df = df.sample(frac=0.1) # select a sample such that getLeastSquares runs in a reasonable amount of time
    # df, _ = getLeastSquares(df, False, False)
    # print(f'Done with least squares: {np.round(time.time()-start_ls,1)}s')
    # col_names = [col for col in df.columns if col.startswith('yInterp')]

    print('Performing dot product')
    ## Convolve the skin spectra with the chemophore spectras
    nmStartEnd_str = ['Wavelength_'+str(nmStart),'Wavelength_'+str(nmEnd)]
    # Data for convolution of form {Output_Column_Name: (chromophore_csv_column_name, chromophore_csv_path), ...}
    dotData = {'HbO2': ('HbO2 cm-1/M','/Users/maycaj/Documents/HSI/Absorbances/HbO2 Absorbance.csv'),
                    'Hb': ('Hb cm-1/M','/Users/maycaj/Documents/HSI/Absorbances/HbO2 Absorbance.csv'),
                    'H2O': ('H2O 1/cm','/Users/maycaj/Documents/HSI_III/Absorbances/Water Absorbance.csv'),
                    'Pheomelanin': ('Pheomelanin cm-1/M','/Users/maycaj/Documents/HSI_III/Absorbances/Pheomelanin.csv'),
                    'Eumelanin': ('Eumelanin cm-1/M','/Users/maycaj/Documents/HSI_III/Absorbances/Eumelanin Absorbance.csv'),
                    'fat':('fat','/Users/maycaj/Documents/HSI_III/Absorbances/Fat Absorbance.csv'),
                    'L':('L','/Users/maycaj/Documents/HSI/Absorbances/LM Absorbance.csv'),
                    'M':('M','/Users/maycaj/Documents/HSI/Absorbances/LM Absorbance.csv'),
                    'S':('S','/Users/maycaj/Documents/HSI/Absorbances/S Absorbance.csv')}
    chromKeys = ['HbO2','Hb']
    for key in chromKeys:
        chrom_dot_spectra(df, nmStartEnd_str, dotData[key][0], key, dotData[key][1], Normalized=True)

    # col_names = ['HbO2','Hb'] # What column names are we training on?

    ## Normalize specified columns within each image (denoted by FloatName)
    # uniqFloats = df['FloatName'].unique()
    # for uniqFloat in uniqFloats: 
    #     # Normalize ratios from 0 to 1 within each image for each columnn
    #     for outputName in ['Hb/HbO2']:
    #         mask = df['FloatName'] == uniqFloat # predictions for one ID
    #         temp_col = df.loc[mask, outputName]
    #         df.loc[mask, outputName] = temp_col - min(temp_col) + 0.001 # adding 0.001 avoids error when there is only one sample in the category
    #         temp_col = df.loc[mask, outputName]
    #         df.loc[mask, outputName] = temp_col / max(temp_col)

    ## FIT MODELS AND SAVE OUTPUT
    TN, FP, FN, TP = 0, 0, 0, 0 # initialize confusion matrix tally
    uniqIDs = df['ID'].unique() # find unique IDs

    # Initialize a dataframe for Location of each classification
    PredLocs = pd.DataFrame([])
    coefs = pd.DataFrame([])

    # Initialize a dataframe for confusion matricies
    confusions = pd.DataFrame(columns=['ID','TN','FP','FN','TP'])

    selectedNums = [int(frac*df.shape[0]) for frac in fracs] # convert the fraction to a number
    # Loop over the selected number of data and each iteration
    for selectedNum in selectedNums:
        for i in range(iterations): # iterate multiple times for error bars
            print(f'Starting iteration: {i}')
            random_state = np.random.randint(0,4294967295) # generate random integer for random state
            dataFrac = df.sample(n=selectedNum,random_state=random_state) # include a fraction of the data 
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state) #shuffle is important to avoid bias
            
            # Run each fold in parallel
            foldOutput = Parallel(n_jobs=n_jobs)(
                delayed(training.process_fold)(foldNum, train_index, test_index, i)
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

    #Add averages to dataframe
    threshold = 0.5 # what fraction of patches of edemaTrue to consider whole leg edemaTrue
    IDavg = IDaccs.drop(['ID',y_col], axis=1).T.mean() #Exclude the non-numerical data, and find mean across IDs using transpose
    IDaccs.insert(2,'ID Avg',IDavg) # Insert ID average
    for thresh in [threshold]: # np.arange(0.3,0.7,0.1): # iterate over possible thresholds
        thresh = np.round(thresh, 1)
        IDroundAvg = IDaccs.drop(['ID',y_col], axis=1).T.apply(lambda cell: postprocessing.roundCells(cell,thresh)).mean() # round cells to threshold (thresh) and then take the mean across the rows
        IDaccs.insert(2,f'ID {thresh} Rounded Avg',IDroundAvg) # add thresholded values as a new column

    # add average across columns
    IDaccs.loc['Column Average'] = IDaccs.drop(['ID',y_col], axis=1).mean(axis=0) 

    # apply 95% CI to iterations data
    hasFolderHasAvg = (IDaccs.loc[:,y_col].notna()) | (IDaccs.index == 'Column Average') # select rows with folder names and the column average
    iteration = IDaccs.loc[hasFolderHasAvg, 'Iter 0 Fold 0':f'Iter {i} Fold {n_splits-1}'] # select all of iteration data
    # iteration = IDaccs.loc[hasFolderHasAvg, 'Iter 0 Fold 0':f'Iter {i} Fold {foldNum}'] # select all of iteration data
    CI95 = iteration.apply(lambda row: postprocessing.bootstrapSeries(row), axis=1)
    IDaccs.insert(3, '95%CI', CI95)
    CI95round = iteration.apply(lambda row: postprocessing.bootstrapSeries(row, round=True), axis=1)
    IDaccs.insert(3, '95%CIround', CI95round)

    # notColAvg = IDaccs.index != 'Column Average'
    yerr = IDaccs.loc[:,'95%CI'].apply(postprocessing.halfDifference) # apply the halfDifference equation to get the error needed for bar chart
    IDaccs.insert(2, 'Yerr', yerr)
    yerrRounded = IDaccs.loc[:,'95%CIround'].apply(postprocessing.halfDifference)
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

    # wholeLeg = columnBarPlot(IDaccs, IDaccs['ID 0.5 Rounded Avg'].notna(),'Labels','ID 0.5 Rounded Avg','YerrRounded',f'ID \n {y_col}','Accuracy',f'{y_col} Leg Acc n={selectedNum} iterations={iterations}')
    patchFig, patchAcc = postprocessing.columnBarPlot(IDaccs, IDaccs['ID Avg'].notna(),'Labels','ID Avg','Yerr',f'ID \n {y_col}','Patch Accuracy',f'{dataConfig}\ny:{y_col} n={selectedNum} iterations={iterations} fracs={fracs} \n data:{filepath.split("/")[-1]}')

    ## Plot the SVM coefficients
    coefFig = plt.figure(figsize=(18,4))
    plt.plot(abs(coefs).median(axis=0),label='Absolute value')
    coef_95CI = coefs.apply(lambda col: postprocessing.bootstrapSeries(col), axis=0)
    filename = filepath.split("/")[-1]
    plt.plot(coef_95CI.iloc[0,:],label='lower')
    plt.plot(coef_95CI.iloc[1,:],label='upper')
    plt.xticks(rotation=90)
    plt.title(f'{dataConfig} data:{filepath.split("/")[-1]} \n iterations: {iterations} n={selectedNum} fracs={fracs}')
    plt.legend()
    plt.tight_layout()

    # Find the confusi on matricies for each ID
    confusion = confusions.groupby('ID').sum()
    confusion = confusion.reset_index()

    # add metadata and save 
    date = filepath.split('/')[-1].split('_')[0] # find date from file name
    IDaccs.loc['Info','Labels'] = f'input filename: {filename}' # add input filename

    ## save CSVs, pdfs, and a copy of the script in the output file
    # meta_row = pd.DataFrame([{col: '' for col in coefs.columns}])
    # meta_row[coefs.columns[0]] = filename
    # coefs = pd.concat([coefs, meta_row], axis=0)
    folder = Path(f'/Users/maycaj/Downloads/SpectrumClassifier2 {str(datetime.now().strftime("%Y-%m-%d %H %M"))}/')
    folder.mkdir(exist_ok=True)
    # IDaccs.to_csv(f'/Users/maycaj/Downloads/{date}IDaccs_n={selectedNum}i={iterations}.csv') # Save the accuracy as csv
    # PredLocs.to_csv(f'/Users/maycaj/Downloads/{date}PredLocs_n={selectedNum}i={iterations}.csv.gz', compression='gzip', index=False) # Save predictions with locations as csv
    # confusion.to_csv(f'/Users/maycaj/Downloads/{date}confusion_n={selectedNum}i={iterations}.csv')
    # confusions.to_csv(f'/Users/maycaj/Downloads/{date}confusions_n={selectedNum}i={iterations}.csv')
    # wholeLeg.savefig(f'/Users/maycaj/Downloads/{date}wholeLeg_n={selectedNum}i={iterations}.pdf')
    patchFig.savefig(folder / f'patch__acc={patchAcc}pct={fracs[0]}i={iterations} {filename}.pdf')
    coefFig.savefig(folder / f'coef__acc={patchAcc}pct={fracs[0]}i={iterations} {filename}.pdf')
    coefs.to_csv(folder / f'coefs__acc={patchAcc}pct={fracs[0]}i={iterations} {filename}', index=False)
    shutil.copy(sys.argv[0], folder / 'SpectrumClassifier2.py')

    plt.show()

    print('All done ;)')
    # breakpoint()
    pass