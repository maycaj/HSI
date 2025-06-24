### Classify each individual pixel from the hyperspectral image using an SVM. The input data is in CSV form. Ouputs are bootstrapped accuracy bar charts and SVM coefficients.
# How to use: Adjust the parameters_dict list at the end of the script and then run. You can add any number of dictionaries to parameters_dict to run a second model after. 

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV
import time
from joblib import Parallel, delayed
import pyarrow.csv as pv
from ChromDotSpectra import chrom_dot_spectra # Make sure you have ChromDotSpectra.py in the same directory
from datetime import datetime
import sys
import os
from pathlib import Path
import shutil
import json  # Add this import for saving parameters as JSON or text
from wakepy import keep
from sklearn.linear_model import SGDClassifier
import gc

class DataLoader:
    @staticmethod
    def load_dataframe(filepath):
        '''
        Args: 
            filepath (str): path to df
        Returns:
            df (DataFrame)
        '''
        print('Loading dataframe...')
        start_time = time.time()
        # table = pv.read_csv(filepath)
        # df = table.to_pandas()
        # table = None # Clear table after use
        df = pd.read_csv(filepath)
        print(f'Done loading dataframe ({np.round(time.time()-start_time,1)}s)')
        return df 

class DataProcessor:
    @staticmethod
    def undersample_class(df, classes, column_label, random_state):
        '''
        Evens out class imbalances via undersampling
        Args:
            df: pandas dataframe 
            classes: string of binary classes to undersample ['class1','class2']
            column_label: string label of the column where undersampling occurs
        Returns:
            undersampled data dataframe
        '''
        df_true = df[df[column_label] == classes[0]]
        df_false = df[df[column_label] == classes[1]]
        min_len = int(min(df_true.shape[0], df_false.shape[0]))
        true_sample = df_true.sample(n=min_len, random_state=random_state)
        false_sample = df_false.sample(n=min_len, random_state=random_state)
        return pd.concat([true_sample, false_sample], axis=0)

    @staticmethod
    def summarize_data(df, uniq_ids, y_col, y_categories):
        '''
        Prints out the sums of the weights and the sums of the samples by Id and by T/F
        Args: 
            df: dataframe 
        '''
        for ID in uniq_ids:
            ID_sum = sum(df['ID'] == ID)
            # print(f'{ID} total: {ID_sum}')
        true_sum = sum(df[y_col] == y_categories[1])
        false_sum = sum(df[y_col] == y_categories[0])
        print(f'{y_categories[0]} total: {false_sum} | {y_categories[1]} total: {true_sum}')

class ModelTrainer:
    @staticmethod
    def optimize_hyperparameters(X_train, y_train):
        '''
        Optimizes hyperparameters for the SVM classifier using GridSearchCV.
        Args:
            X_train: Training features
            y_train: Training labels
        Returns:
            best_params: Best hyperparameters found
        '''
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }
        grid_search = GridSearchCV(SVC(), param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        print(f"Best parameters: {grid_search.best_params_}")
        return grid_search.best_params_

    @staticmethod
    def process_fold(fold_num, train_index, test_index, i, uniq_ids, data_frac, y_categories, y_col, col_names, scale, optimize, stochastic, random_state):
        '''
        Processes each fold of the data. Is wrapped into a function so that each fold can be parallelized
        Args:
            fold_num: what number fold we are on
            train_index: index of the training examples
            test_index: index of the testing index
            i: what number iteration we are on
        
        Returns: 
            confusionFold: confusion matricies for this fold
            PredLocFold: predictions and their locations for this fold
            IDaccFold: ID accuracies for this fold
            coefFold: SVM coefficients for this fold
        '''
        train_ids = uniq_ids[train_index]
        test_ids = uniq_ids[test_index]
        print(f'Fold {fold_num} Train IDs: {train_ids}')
        print(f'Fold {fold_num} Test IDs: {test_ids}')
        train = data_frac[data_frac['ID'].isin(train_ids)]
        test = data_frac[data_frac['ID'].isin(test_ids)].copy()
        train = DataProcessor.undersample_class(train, y_categories, y_col, random_state)

        DataProcessor.summarize_data(train, uniq_ids, y_col, y_categories)

        X_train = train.loc[:, col_names] 
        y_train = train[y_col]
        X_test = test.loc[:, col_names]
        y_test = test[y_col]

        if optimize: 
            svc_params = ModelTrainer.optimize_hyperparameters(X_train, y_train)
            svm = SVC(**svc_params)
        else:
            if stochastic:
                svm = SGDClassifier(loss='hinge', max_iter=1000, tol=1e-3)
            else:
                svm = SVC(kernel='linear')

        if scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
        svm.fit(X_train, y_train)

        if scale:
            X_test = scaler.transform(X_test)
        y_pred = svm.predict(X_test)

        if optimize:
            if svc_params['kernel'] == 'linear':
                coef_fold = pd.DataFrame(svm.coef_[0].reshape(1, -1), columns=col_names)
            elif svc_params['kernel'] == 'rbf':
                coef_fold = pd.DataFrame(np.zeros((1, len(col_names))), columns=col_names)
        else:
            coef_fold = pd.DataFrame(svm.coef_[0].reshape(1, -1), columns=col_names)

        test.loc[:, 'correct'] = y_test == y_pred
        test.loc[:, 'yPred'] = y_pred
        test.loc[:, 'yTest'] = y_test

        confusion_fold = pd.DataFrame([])
        uniq_test_ids = test['ID'].unique()
        for ID in uniq_test_ids:
            test_id = test[(test['ID'] == ID)] 
            if not test_id.empty:
                cm = confusion_matrix(test_id['yTest'], test_id['yPred'], labels=y_categories)
                if cm.size != 1:
                    TP, FN, FP, TN = cm.ravel()
                    confusion_fold = pd.concat([confusion_fold, pd.DataFrame([[fold_num, ID, TN, FP, FN, TP]], columns=['foldNum', 'ID', 'TN', 'FP', 'FN', 'TP'])])

        pred_loc_fold = test[['ID', 'yPred', 'FloatName', 'correct', y_col]].copy()
        pred_loc_fold['foldNum'] = fold_num

        print(f'Patch Acc Fold {fold_num} iteration {i}: {np.mean(test["correct"])}')

        ID_acc_fold = test.groupby(['ID', y_col])['correct'].mean()
        ID_acc_fold = ID_acc_fold.to_frame(name=f'Iter {i} Fold {fold_num}')
        ID_acc_fold.reset_index(inplace=True)

        return confusion_fold, pred_loc_fold, ID_acc_fold, coef_fold

class PostProcessor:
    @staticmethod
    def round_cells(cell, thresh):
        '''
        Rounds cells to desired threshold
        Args:
            cell: cell of a dataframe
            thresh: threshold decimal
        Returns:
            output: rounded cells
        '''
        return np.where(np.isnan(cell), np.nan, np.where(cell > thresh, 1, 0))

    @staticmethod
    def bootstrap_series(series, round=False):
        '''
        Bootstraps to make 95% CI across a pandas dataframe row
        Args:
            series: dataframe row
        Returns:
            lower, upper: Confidence interval bounds
        '''
        confidence = 0.95
        bootstrapped_means = []
        series = series[series.notna()].values
        if round:
            series = np.round(series)
        for _ in range(1000):
            sample = np.random.choice(series, size=len(series), replace=True)
            bootstrapped_means.append(np.mean(sample))
        lower = np.percentile(bootstrapped_means, (1-confidence)*100/2)
        upper = np.percentile(bootstrapped_means, (1+confidence)*100/2)
        return np.array([np.round(float(lower), 2), np.round(float(upper), 2)])

    @staticmethod
    def half_difference(input):
        '''
        Construct yerror 
        Args:
            input: [lowerCI, upperCI]
        Returns:
            half of difference between lower CI and upper CI
        '''
        return np.abs(input[0] - input[1]) / 2
    
    @staticmethod
    def get_yerr(df_CI, df_avg):
        '''
        Gets the asymmetric lengths of the error bars
        Args:
            df_CI: the bootstrapped confidence intervals 
            df_avg: the average accuracy
        '''
        df_CI = np.vstack(df_CI)
        lowers = np.abs(df_avg.values - df_CI[:,0])
        uppers = np.abs(df_avg.values - df_CI[:,1])
        return lowers, uppers

    @staticmethod
    def column_bar_plot(df, has_values, labels_key, values_key, error_key_lower, error_key_upper, xlabel, ylabel, title):
        '''
        Plots down a column, creates bar chart with error bars
        Args:
            df: dataFrame
            has_values: Key for Dataframe column. The column's rows with values will be used in this analysis
            labels_key: Key for labels column in Dataframe
            values_key: Key for values column in Dataframe
            error_key_lower: lower length of error bar
            error_key_upper: upper length of error bar
            xlabel, ylabel, title: for plot
        '''
        labels = df[has_values][labels_key]
        labels.loc['Column Average'] = 'Column Average'
        labels = labels.astype(str)
        values = df[has_values][values_key]
        # yerr_plot = df[has_values][error_key]
        # yerr_plot = np.vstack(yerr_plot)#.T
        # yerr_lower = yerr_plot[:,0]
        # yerr_upper = yerr_plot[:,1]
        yerr_lower = df[error_key_lower]
        yerr_upper = df[error_key_upper]
        fig, ax = plt.subplots()
        bars = ax.bar(labels, values, yerr=[yerr_lower,yerr_upper])
        ax.tick_params(axis='x', labelsize=5, rotation=90)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        acc = str(round(values['Column Average']*100, 1))
        plt.title(title + ' Average=' + acc)
        plt.tight_layout()
        return fig, acc

class SpectrumClassifier:
    def __init__(self, run_num, save_folder, filepath, fracs, iterations, n_jobs, y_col, scale, optimize, stochastic, nm_start, nm_end, data_config, chrom_keys, d65, scrambled_chrom):
        self.run_num = run_num
        self.save_folder = save_folder
        self.filepath = filepath
        self.fracs = fracs
        self.iterations = iterations
        self.n_jobs = n_jobs
        self.y_col = y_col
        self.scale = scale
        self.optimize = optimize
        self.stochastic = stochastic
        self.nm_start = nm_start
        self.nm_end = nm_end
        self.data_config = data_config
        self.chrom_keys = chrom_keys
        self.d65 = d65
        self.scrambled_chrom = scrambled_chrom
        self.start_time = time.time()
        self.df = DataLoader.load_dataframe(filepath)

    def process_data(self):
        '''
        Processes the data, applies configurations, and prepares for training
        Args:
            chrom_keys: chromophore keys to be fit on. If not none, chrom_keys are fit on instead of the wavelength columns.
        '''
        print('Selecting IDs')
        data_config = self.data_config
        data_configs = {
            'Round 1: cellulitis or edemafalse': (11, 12, 15, 18, 20, 22, 23, 26, 34, 36), 
            'Round 1: peripheral or edemafalse': (1, 2, 5, 6, 7, 8, 9, 10, 13, 14, 19, 21, 24, 27, 29, 30, 31, 32, 33, 35, 37, 38, 39, 40),
            'Round 1 & 2: cellulitis or edemafalse': (11, 12, 15, 18, 20, 22, 23, 26, 34, 36, 45, 59, 61, 70),
            'Round 1 & 2: peripheral or edemafalse': (1, 2, 5, 6, 7, 8, 9, 10, 13, 14, 19, 21, 24, 27, 29, 30, 31, 32, 33, 35, 37, 38, 39, 40, 41, 42, 43, 46, 47, 48, 49, 51, 53, 54, 55, 56, 57, 58, 60, 62, 63, 64, 65, 66, 67, 68, 69, 71, 72)
        }
        self.df = self.df[(self.df['ID'].isin(data_configs[data_config]) | (self.df['Foldername'] == 'EdemaFalse'))] # Keep all of EdemaFalse Data
        self.df = self.df[self.df['ID'] != 0] # Remove Dr. Pare
        self.n_splits = len(self.df['ID'].unique()) # Make the number of splits be the same as the number of IDs (patients)

        self.df.rename( # rename all of the 'Wavelength_' columns to be numberical floats
            columns={col: float(col.split('Wavelength_')[1])
                for col in self.df.columns if col.startswith('Wavelength_')
        }, inplace=True)
        self.col_names = [col for col in self.df.columns if type(col) == float] # Select the columns that are floats
        self.col_names = [col for col in self.col_names if (col > self.nm_start) & (col < self.nm_end)] # select within specified range
        # self.col_names = [col for col in self.df.columns if col.startswith('Wavelength_')] # column names for X
        # self.col_names = [float(col.split('Wavelength_')[1]) for col in self.col_names] 
        # self.col_names = [col for col in self.col_names if (float(col.split('_')[1]) >= self.nm_start) & (float(col.split('_')[1]) <= self.nm_end)]

        self.y_categories = self.df[self.y_col].unique()
        print('Performing dot product')
        # nm_start_end_str = ['Wavelength_' + str(self.nm_start), 'Wavelength_' + str(self.nm_end)]
        dot_data = { # Name, column key, and filepath of the absorbance data
            'HbO2': ('HbO2 cm-1/M', '/Users/maycaj/Documents/HSI/Absorbances/HbO2 Absorbance.csv'),
            'Hb': ('Hb cm-1/M', '/Users/maycaj/Documents/HSI/Absorbances/HbO2 Absorbance.csv'),
            'H2O': ('H2O 1/cm', '/Users/maycaj/Documents/HSI/Absorbances/Water Absorbance.csv'),
            'Pheomelanin': ('Pheomelanin cm-1/M', '/Users/maycaj/Documents/HSI/Absorbances/Pheomelanin.csv'),
            'Eumelanin': ('Eumelanin cm-1/M', '/Users/maycaj/Documents/HSI/Absorbances/Eumelanin Absorbance.csv'),
            'fat': ('fat', '/Users/maycaj/Documents/HSI/Absorbances/Fat Absorbance.csv'),
            'L': ('L', '/Users/maycaj/Documents/HSI/Absorbances/LM Absorbance.csv'),
            'M': ('M', '/Users/maycaj/Documents/HSI/Absorbances/LM Absorbance.csv'),
            'S': ('S', '/Users/maycaj/Documents/HSI/Absorbances/S Absorbance.csv')
        }
        if self.chrom_keys is not None:
            for key in self.chrom_keys:
                chrom_dot_spectra(self.df, [self.nm_start,self.nm_end], dot_data[key][0], key, dot_data[key][1],
                                   self.d65, self.scrambled_chrom, normalized=True, plot=False)
            self.col_names = self.chrom_keys # replace column names 
        self.TN, self.FP, self.FN, self.TP = 0, 0, 0, 0
        self.uniq_ids = np.sort(self.df['ID'].unique())

        # Initialize dataframes for data processing
        self.PredLocs = pd.DataFrame([])
        self.coefs = pd.DataFrame([])
        self.confusions = pd.DataFrame(columns=['ID', 'TN', 'FP', 'FN', 'TP'])
        self.selected_num = int(self.fracs * self.df.shape[0])

    def train_and_evaluate(self):
        '''
        Trains the model and evaluates it using cross-validation
        '''
        for i in range(self.iterations):
            print(f'Starting iteration: {i}')
            random_state = np.random.randint(0, 4294967295)
            data_frac = self.df.sample(n=self.selected_num, random_state=random_state) # sample a subset of df
            while  not np.array_equal(self.uniq_ids, np.sort(data_frac['ID'].unique())): # Resample if the fraction has zero examples for any class
                data_frac = self.df.sample(n=self.selected_num, random_state=random_state)
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=random_state)
            fold_output = Parallel(n_jobs=self.n_jobs)( # process each fold in parallel
                delayed(ModelTrainer.process_fold)(fold_num, train_index, test_index, i, self.uniq_ids, data_frac, 
                                                   self.y_categories, self.y_col, self.col_names, self.scale, 
                                                   self.optimize, self.stochastic, random_state)
                for fold_num, (train_index, test_index) in enumerate(kf.split(self.uniq_ids))
            )

            # joblib requires concatenation of the outputs 
            confusion_fold, pred_loc_fold, ID_acc_fold, coef_fold = zip(*fold_output)
            confusion_fold = pd.concat(confusion_fold, ignore_index=True)
            pred_loc_fold = pd.concat(pred_loc_fold, ignore_index=True)
            ID_acc_fold = pd.concat(ID_acc_fold, ignore_index=True)
            coef_fold = pd.concat(coef_fold, ignore_index=True)
            self.confusions = pd.concat([self.confusions, confusion_fold])
            self.PredLocs = pd.concat([self.PredLocs, pred_loc_fold])
            self.coefs = pd.concat([self.coefs, coef_fold])
            if i == 0:
                self.IDaccs = ID_acc_fold
            else:
                self.IDaccs = self.IDaccs.merge(ID_acc_fold, on=['ID', self.y_col], how='left')
        end_time = time.time()
        print(f'Done with Training \n Total time: {end_time - self.start_time}')

    def save_results(self):
        '''
        Saves results, plots, and metadata
        '''
        # Make a dataframe ID accs with each ID and their corresponding accuracies; Apply bootstrapping to create a 95% confidence interval
        threshold = 0.5
        ID_avg = self.IDaccs.drop(['ID', self.y_col], axis=1).T.mean()
        self.IDaccs.insert(2, 'ID Avg', ID_avg)
        for thresh in [threshold]:
            thresh = np.round(thresh, 1)
            ID_round_avg = self.IDaccs.drop(['ID', self.y_col], axis=1).T.apply(lambda cell: PostProcessor.round_cells(cell, thresh)).mean()
            self.IDaccs.insert(2, f'ID {thresh} Rounded Avg', ID_round_avg) # add thresholded values as a new column
        self.IDaccs.loc['Column Average'] = self.IDaccs.drop(['ID', self.y_col], axis=1).mean(axis=0) # add average across columns
        has_folder_has_avg = (self.IDaccs.loc[:, self.y_col].notna()) | (self.IDaccs.index == 'Column Average')
        iteration = self.IDaccs.loc[has_folder_has_avg, 'Iter 0 Fold 0':f'Iter {self.iterations-1} Fold {self.n_splits-1}']

        # Bootstrap rows and add to dataframe
        CI95 = iteration.apply(lambda row: PostProcessor.bootstrap_series(row), axis=1) # apply 95% CI to iterations data
        self.IDaccs.insert(3, '95%CI', CI95)
        CI95round = iteration.apply(lambda row: PostProcessor.bootstrap_series(row, round=True), axis=1)
        self.IDaccs.insert(3, '95%CIround', CI95round)

        # Add errors needed for bar charts 
        # yerr = self.IDaccs.loc[:, '95%CI'].apply(PostProcessor.half_difference) # apply the halfDifference equation to get the error needed for bar chart
        self.IDaccs.insert(2,'Yerr', np.abs(self.IDaccs.loc[:, '95%CI'] - self.IDaccs.loc[:, 'ID Avg']))
        self.IDaccs.insert(2,'Yerr lower', np.vstack(self.IDaccs['Yerr'])[:,0])
        self.IDaccs.insert(3, 'Yerr upper', np.vstack(self.IDaccs['Yerr'])[:,1])
        # yerr_rounded = self.IDaccs.loc[:, '95%CIround'].apply(PostProcessor.half_difference)
        self.IDaccs.insert(2,'Yerr round', np.abs(self.IDaccs.loc[:, '95%CIround'] - self.IDaccs.loc[:, f'ID {thresh} Rounded Avg']))
        self.IDaccs.insert(2,'Yerr lower round', np.vstack(self.IDaccs['Yerr round'])[:,0])
        self.IDaccs.insert(3, 'Yerr upper round', np.vstack(self.IDaccs['Yerr round'])[:,1])
        # self.IDaccs.insert(2, 'YerrRounded', yerr_rounded)

        # Add labels needed for bar charts
        self.IDaccs.insert(2, 'Labels', 'ID: ' + self.IDaccs['ID'].astype(str) + '\n' + 'Cat: ' + self.IDaccs[self.y_col].astype(str)) # find values to plot in bar chart
        self.IDaccs.loc['Column Average', 'Labels'] = 'Column Average'
        Folder_acc = self.IDaccs.groupby(self.y_col)['ID Avg'].mean() # find accuracy by foldernam; add to ID accs
        Folder_acc = Folder_acc.reset_index(inplace=False)
        Folder_acc['Labels'] = Folder_acc[self.y_col]
        self.IDaccs = pd.concat([self.IDaccs, Folder_acc], axis=0)
        Folder_round = self.IDaccs.groupby(self.y_col)['ID 0.5 Rounded Avg'].mean() # find rounded accuracy by foldername; add to ID_accs
        Folder_round = Folder_round.reset_index(inplace=False)
        self.IDaccs.loc[self.IDaccs['Labels'] == self.y_categories[1], 'ID 0.5 Rounded Avg'] = Folder_round.loc[Folder_round[self.y_col] == self.y_categories[1], 'ID 0.5 Rounded Avg']
        self.IDaccs.loc[self.IDaccs['Labels'] == self.y_categories[0], 'ID 0.5 Rounded Avg'] = Folder_round.loc[Folder_round[self.y_col] == self.y_categories[0], 'ID 0.5 Rounded Avg']

        # Plot IDaccs as a barchart with error bars
        patch_fig, patch_acc = PostProcessor.column_bar_plot(self.IDaccs, self.IDaccs['ID Avg'].notna(), 'Labels',
                                                              'ID Avg', 'Yerr lower', 'Yerr upper', f'ID \n {self.y_col}', 'Patch Accuracy',
                                                                f'{self.data_config}\ny:{self.y_col} n={self.selected_num} iterations={self.iterations} fracs={self.fracs} \n data:{self.filepath.split("/")[-1]}')
        leg_fig, _ = PostProcessor.column_bar_plot(self.IDaccs, self.IDaccs[f'ID {thresh} Rounded Avg'].notna(), 'Labels',
                                                              f'ID {thresh} Rounded Avg', 'Yerr lower round', 'Yerr upper round', f'ID \n {self.y_col}', 'Rounded Accuracy',
                                                                f'{self.data_config}\ny:{self.y_col} n={self.selected_num} iterations={self.iterations} fracs={self.fracs} \n data:{self.filepath.split("/")[-1]}')

        # Plot the SVM coefficients
        coef_fig = plt.figure(figsize=(18, 4)) 
        coef_95CI = self.coefs.apply(lambda col: PostProcessor.bootstrap_series(col), axis=0)
        filename = self.filepath.split("/")[-1]
        plt.plot(abs(self.coefs.median(axis=0)), label='abs(median)')
        plt.plot(coef_95CI.iloc[0, :], label='lower')
        plt.plot(coef_95CI.iloc[1, :], label='upper')
        plt.xticks(rotation=90)
        plt.title(f'{self.data_config} data:{self.filepath.split("/")[-1]} \n iterations: {self.iterations} n={self.selected_num} fracs={self.fracs}')
        plt.legend()
        plt.tight_layout()

        # find the confusion matricies for each ID
        confusion = self.confusions.groupby('ID').sum() 
        confusion = confusion.reset_index()

        # Save all of the dataframes, figures, and this script into a new folder for each model run
        folder = self.save_folder / f'Run {self.run_num}'
        folder.mkdir(parents=True, exist_ok=True)

        # Save parameters as a .txt file
        parameters_file = folder / 'parameters.txt'
        with open(parameters_file, 'w') as f:
            json.dump({
                'run_num': self.run_num,
                'save_folder': str(self.save_folder),
                'filepath': self.filepath,
                'fracs': self.fracs,
                'iterations': self.iterations,
                'n_jobs': self.n_jobs,
                'y_col': self.y_col,
                'scale': self.scale,
                'optimize': self.optimize,
                'stochastic': self.stochastic,
                'nm_start': self.nm_start,
                'nm_end': self.nm_end,
                'data_config': self.data_config,
                'chrom_keys': self.chrom_keys,
                'd65': self.d65
            }, f, indent=4)

        self.IDaccs.loc['Info', 'Labels'] = f'input filename: {filename}' # add metadata
        # IDaccs.to_csv(f'/Users/maycaj/Downloads/{date}IDaccs_n={selectedNum}i={iterations}.csv') # Save the accuracy as csv
        # PredLocs.to_csv(f'/Users/maycaj/Downloads/{date}PredLocs_n={selectedNum}i={iterations}.csv.gz', compression='gzip', index=False) # Save predictions with locations as csv
        # confusion.to_csv(f'/Users/maycaj/Downloads/{date}confusion_n={selectedNum}i={iterations}.csv')
        # confusions.to_csv(f'/Users/maycaj/Downloads/{date}confusions_n={selectedNum}i={iterations}.csv')
        leg_fig.savefig(folder / f'leg__pct={self.fracs}i={self.iterations} {filename}.pdf')
        patch_fig.savefig(folder / f'patch__acc={patch_acc}pct={self.fracs}i={self.iterations} {filename}.pdf')
        coef_fig.savefig(folder / f'coef__acc={patch_acc}pct={self.fracs}i={self.iterations } {filename}.pdf')
        self.coefs.to_csv(folder / f'coefs__acc={patch_acc}pct={self.fracs}i={self.iterations} {filename}', index=False)
        shutil.copy(sys.argv[0], folder / 'SpectrumClassifier2.py') 
        print('All done ;)')

    def run(self):
        '''
        Main method to execute the entire pipeline
        '''
        self.process_data()
        self.train_and_evaluate()
        self.save_results()

if __name__ == '__main__':
    # Can run several different sets of parameters in sucession 
    paramters_dict = [
        {'fracs':1, # Fraction of examples to include
        'filepath':'/Users/maycaj/Documents/HSI/PatchCSVs/May_29_NOCR_FullRound1and2AllWLs_medians.csv', # filepath of our dataset
        'iterations':150, # number of iterations to run the model
        'n_jobs':-1, # 1 is for running normally, any number above 1 is for parallelization, and -1 takes all of the available CPUs
        'y_col':'Foldername', # what column of the data from filepath to fit on
        'scale':False, # if using StandardScaler() for the SVM
        'optimize':False, # if optimizing, C, kernel, and gamma 
        'stochastic':False, # if using stochastic gradient descent (better for larger amounts of data). Else use linear SVM 
        'nm_start':451.18, # the minimum wavelength to include in model fits
        'nm_end':954.83, # the maximum wavelength to include in model fits
        'data_config' :  'Round 1 & 2: cellulitis or edemafalse', # which rounds and which disease group to fit on. Check data_configs for options
        'chrom_keys': ['HbO2', 'Hb', 'H2O', 'Pheomelanin', 'Eumelanin', 'fat', 'L', 'M', 'S'], # None if using the camera wavelenbths, else using the dot product of the skin chromophore as the features to fit on. See the keys of dot_data for chromophore options.
        'd65': False, # If scaling the data by the daylight axis (d65). Used in conjunction with ['L','M','S'] as chrom_keys
        'scrambled_chrom': True}, # If scrambling the chromophores to be the same size but completely random

    ]
    save_folder = Path(f'/Users/maycaj/Downloads/SpectrumClassifier2 {str(datetime.now().strftime("%Y-%m-%d %H %M"))}')
    with keep.running(on_fail='warn'): # keeps running when lid is shut
        for run_num, parameters in enumerate(paramters_dict):
            classifier = SpectrumClassifier(run_num, save_folder, **parameters)
            classifier.run()
    plt.show()