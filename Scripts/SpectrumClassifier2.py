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
from datetime import datetime
import sys
from pathlib import Path
import shutil
import json  # Add this import for saving parameters as JSON or text
from sklearn.linear_model import SGDClassifier
import plotly.graph_objects as go
# import dask.dataframe as dd


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
        # table = pv.read_csv(filepath)/
        # df = table.to_pandas()
        # table = None # Clear table after use
        df = pd.read_csv(filepath)
        print(f'Done loading dataframe ({np.round(time.time()-start_time,1)}s)')
        df_medians = df.groupby(['ID','Foldername','FloatName']).median()
        df_medians.reset_index(inplace=True)
        return df 
    
class ChromDotSpectra:
    @staticmethod
    def load_and_interpolate(file_path, wavelength_column, value_column, x_interp):
        """
        Load a CSV file and interpolate the values based on the given wavelength column and value column.
        
        Parameters:
            file_path (str): Path to the CSV file.
            wavelength_column (str): Column name for wavelengths.
            value_column (str): Column name for the values to interpolate.
            x_interp (array-like): Wavelengths to interpolate to.
        
        Returns:
            x_interp: wavelength values that correspond with y_interp
            y_interp: yAbs values interpolated to match with the x_interp
            x_abs: x original absorbance values (wavelength in nm)
            y_abs: y original absorbance values 
            range: [min_x_abs, max_x_abs]
        """
        csv_data = pd.read_csv(file_path)
        if not csv_data[wavelength_column].is_monotonic_increasing:
            csv_data = csv_data.sort_values(by=wavelength_column)
        x_abs = csv_data[wavelength_column].values
        y_abs = csv_data[value_column].values
        y_abs = y_abs / max(y_abs) # Scale from 0 to 1
        y_interp = np.interp(x_interp, x_abs, y_abs)
        if max(y_interp) != 0: 
            y_interp = y_interp / max(y_interp)
        return x_interp, y_interp, x_abs, y_abs

    @staticmethod
    def chrom_dot_spectra(data, nmStartEnd, spectraName, outputName, absorbPath, d65_and_cones=True, scrambled_chrom=False, normalized=False, plot=True):
        '''
        Takes dataframe and performs dot product with Hemoglobin's response curve to see the total absorbance of hemoglobin in the skin.
        Args:
            data: main pandas dataframe with wavelength and demographic data
            nmStartEnd: [startingWavelength, endingWavelength] in data
            spectraName: Name of column in the absorbance csv
            absorbPath: path to .csv with the absorbances
            d65 (bool): if combining the spectra with the daylight axis
            normalized (bool): if dividing the resultant output by its max such that the highest number is 1
            plot (bool): if plotting
        '''
        skin_waves = list(data.loc[:,nmStartEnd[0]:nmStartEnd[1]].columns) # find the wavelengths captured by camera

        # find the wavelengths where the skin wavelengths, absorbance wavelengths, and (if applicable) d65 wavelengths overlap
        x_abs = pd.read_csv(absorbPath)['lambda nm'].values
        dot_waves = [col for col in skin_waves if (col >= min(x_abs)) & (col <= max(x_abs))]
        d65_path = '/Users/cameronmay/Documents/HSI/Absorbances/CIE_std_illum_D65.csv'
        if d65_and_cones: # if d65, ensure that the wavelengths are within the range of d65
            x_abs_d65 = pd.read_csv(d65_path)['lambda nm'].values
            dot_waves = [col for col in dot_waves if (col >= min(x_abs_d65)) & (col <= max(x_abs_d65))]
        
        x_interp, y_interp_chrom, x_abs, y_abs = ChromDotSpectra.load_and_interpolate(absorbPath,'lambda nm',spectraName,dot_waves)

        if normalized: #normalize so that the AUC is consistent across each chromophore. Normalization does not matter at all if you are using StandardScaler()
            y_interp_area = np.trapezoid(y_interp_chrom)
            y_interp_chrom = y_interp_chrom / y_interp_area

        if scrambled_chrom: # if chromophores need to be scrambled
            # y_interp_chrom = np.random.rand(np.size(y_interp_chrom)) # Completely random values
            np.random.shuffle(y_interp_chrom) # Randomized order of chromophore only

        if plot:
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=x_interp, y=y_interp_chrom, name=f'Interp {spectraName}'))
            fig1.add_trace(go.Scatter(x=x_abs, y=y_abs, name=f'Original {spectraName}'))
            fig1.show()

        # find wavelengths that are in range for both the chromophores and the skin
        data_selected = data.loc[:,dot_waves].values
        if d65_and_cones: # multiply by daylight axis
            x_interp, y_interp_d65, x_abs_d65, y_abs_d65 = ChromDotSpectra.load_and_interpolate(d65_path,'lambda nm','D65',dot_waves)
            if plot:
                fig1.add_trace(go.Scatter(x=x_interp, y=y_interp_d65, name=f'Interp d65'))
                fig1.add_trace(go.Scatter(x=x_abs_d65, y=y_abs_d65, name=f'Original d65'))
                fig1.show()
            data_selected = data_selected * y_interp_d65 
        else: # if not d65, convert to apparent absorbance
            data_selected = np.where(data_selected == 0, 1e-10, data_selected) # Ensure there is no divide by zero error in next step
            data_selected = np.log10(1 / data_selected) # Absorbance = log10(1 / reflectance)
        data[outputName] = np.dot(data_selected, y_interp_chrom)

        # Normalize output
        # if normalized: 
            # data[outputName] = data[outputName] - data[outputName].min() + 0.01 # Doing this gets rid of a lot of the ability to segment the veins and also messed with decoding accuracy. It is also not realistic because there is still absorbance occuring even at the raw data's lowest point.
            # data[outputName] = data[outputName] / data[outputName].max() # Normalizing here is a data leak because it is before the train_test_split

        output_dict = {wavelength: [value] for wavelength, value in zip(x_interp, y_interp_chrom)}
        output_dict['label'] = [outputName]
        chrom_interp = pd.DataFrame(output_dict)
        return chrom_interp

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
            uniq_ids: unique IDs in our dataset
            y_col: the column we are fitting on
            y_categories: the binary categories that we are fitting on
        '''
        # for ID in uniq_ids:
        #     ID_sum = sum(df['ID'] == ID)
            # print(f'{ID} total: {ID_sum}')
        true_sum = sum(df[y_col] == y_categories[1])
        false_sum = sum(df[y_col] == y_categories[0])
        print(f'{y_categories[0]} total: {false_sum} | {y_categories[1]} total: {true_sum}')



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
        bars = ax.bar(labels, values, yerr=[yerr_lower,yerr_upper], color='grey')
        ax.tick_params(axis='x', labelsize=5, rotation=90)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
    
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        acc = str(round(values['Column Average']*100, 1))
        plt.title(title + ' Average=' + acc)
        plt.tight_layout()
        return fig, acc

class SpectrumClassifier:
    def __init__(self, run_num, save_folder, filepath, fracs, iterations, n_jobs, y_col, scale, optimize, stochastic, nm_start, nm_end, data_config, chrom_keys, d65_and_cones, scrambled_chrom, apparent_abs):
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
        self.nm_start = nm_start # ___ round to closest camera wavekength 
        self.nm_end = nm_end
        self.data_config = data_config
        self.chrom_keys = chrom_keys
        self.d65_and_cones = d65_and_cones
        self.scrambled_chrom = scrambled_chrom
        self.start_time = time.time()
        self.df = DataLoader.load_dataframe(filepath)
        self.apparent_abs = apparent_abs

    def process_data(self):
        '''
        Processes the data, applies configurations, and prepares for training
        '''
        print('Selecting IDs')
        data_config = self.data_config
        data_configs = {
            'Round 1: cellulitis or edemafalse': (11, 12, 15, 18, 20, 22, 23, 26, 34, 36), 
            'Round 1: peripheral or edemafalse': (1, 2, 5, 6, 7, 8, 9, 10, 13, 14, 19, 21, 24, 27, 29, 30, 31, 32, 33, 35, 37, 38, 39, 40),
            'Round 1 & 2: cellulitis or edemafalse': (11, 12, 15, 18, 20, 22, 23, 26, 34, 36, 45, 59, 61, 70),
            'Round 1 & 2: peripheral or edemafalse': (1, 2, 5, 6, 7, 8, 9, 10, 13, 14, 19, 21, 24, 27, 29, 30, 31, 32, 33, 35, 37, 38, 39, 40, 41, 42, 43, 46, 47, 48, 49, 51, 53, 54, 55, 56, 57, 58, 60, 62, 63, 64, 65, 66, 67, 68, 69, 71, 72),
            'Round 1, 2, & 3: cellulitis/edemafalse + controls': (11, 12, 15, 18, 20, 22, 23, 26, 34, 36, 45, 59, 61, 70, 73, 76, 78, 83, 84, 85, 86, 88, 89, 90, 91),
            'Round 1, 2, & 3: peripheral/edemafalse + controls': (1, 2, 5, 6, 7, 8, 9, 10, 13, 14, 19, 21, 24, 27, 29, 30, 31, 32, 33, 35, 37, 38, 39, 40, 41, 42, 43, 46, 47, 48, 49, 51, 53, 54, 55, 56, 57, 58, 60, 62, 63, 64, 65, 66, 67, 68, 69, 71, 72, 74, 75, 77, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90, 91), 
        }
        self.df = self.df[(self.df['ID'].isin(data_configs[data_config]) | (self.df['Foldername'] == 'EdemaFalse'))] # Keep all of EdemaFalse Data
        self.df = self.df[self.df['ID'] != 0] # Remove Dr. Pare
        self.n_splits = len(self.df['ID'].unique()) # Make the number of splits be the same as the number of IDs (patients)

        # map the wavelengths to the closest camera wavelength
        camera_nm = [376.61, 381.55, 386.49, 391.43, 396.39, 401.34, 406.3, 411.27, 416.24, 421.22, 426.2, 431.19, 436.18, 441.17, 446.17, 451.18, 456.19, 461.21, 466.23, 471.25, 476.28, 481.32, 486.36, 491.41, 496.46, 501.51, 506.57, 511.64, 516.71, 521.78, 526.86, 531.95, 537.04, 542.13, 547.23, 552.34, 557.45, 562.57, 567.69, 572.81, 577.94, 583.07, 588.21, 593.36, 598.51, 603.66, 608.82, 613.99, 619.16, 624.33, 629.51, 634.7, 639.88, 645.08, 650.28, 655.48, 660.69, 665.91, 671.13, 676.35, 681.58, 686.81, 692.05, 697.29, 702.54, 707.8, 713.06, 718.32, 723.59, 728.86, 734.14, 739.42, 744.71, 750.01, 755.3, 760.61, 765.92, 771.23, 776.55, 781.87, 787.2, 792.53, 797.87, 803.21, 808.56, 813.91, 819.27, 824.63, 830.0, 835.37, 840.75, 846.13, 851.52, 856.91, 862.31, 867.71, 873.12, 878.53, 883.95, 889.37, 894.8, 900.23, 905.67, 911.11, 916.56, 922.01, 927.47, 932.93, 938.4, 943.87, 949.35, 954.83, 960.31, 965.81, 971.3, 976.8, 982.31, 987.82, 993.34, 998.86, 1004.39, 1009.92, 1015.45, 1020.99, 1026.54, 1032.09, 1037.65, 1043.21]
        self.nm_start = min(camera_nm, key = lambda x: abs(x - self.nm_start))
        self.nm_end = min(camera_nm, key = lambda x: abs(x-self.nm_end))

        # Find the columns used for training 
        self.df.rename( # rename all of the 'Wavelength_' columns to be numberical floats
            columns={col: float(col.split('Wavelength_')[1])
                for col in self.df.columns if col.startswith('Wavelength_')
        }, inplace=True)
        self.col_names = [col for col in self.df.columns if type(col) == float] # Select the columns that are floats
        self.col_names = [col for col in self.col_names if (col > self.nm_start) & (col < self.nm_end)] # select within specified range
        # self.col_names = [col for col in self.df.columns if col.startswith('Wavelength_')] # column names for X
        # self.col_names = [float(col.split('Wavelength_')[1]) for col in self.col_names] 
        # self.col_names = [col for col in self.col_names if (float(col.split('_')[1]) >= self.nm_start) & (float(col.split('_')[1]) <= self.nm_end)]

        # Initialize dataframes for data processing
        self.PredLocs = pd.DataFrame([])
        self.coefs = pd.DataFrame([])
        self.selected_num = int(self.fracs * self.df.shape[0])
        self.y_categories = self.df[self.y_col].unique()
        self.confusions = pd.DataFrame(columns=['ID', 'TN', 'FP', 'FN', 'TP'])
        self.TN, self.FP, self.FN, self.TP = 0, 0, 0, 0
        self.uniq_ids = np.sort(self.df['ID'].unique())
        self.chrom_interp = pd.DataFrame([])

        # Do dot product if there are chromophores in chrom_keys
        self.chrom_interps = self.dot_product()
        self.apparent_absorbance()

    def apparent_absorbance(self):
        '''
        Converts reflectance data into apparent absorbance
        '''
        if self.apparent_abs == True: # Convert to apparent absorbance. A = -log10(R)
            print('Finding apparent_absorbance...')
            self.df[self.col_names] = -self.df[self.col_names].apply(np.log10)

    def dot_product(self):
        '''
        Takes the dot product of the spectra with the relevant chromophores. Adds columns to df defined by chrom_keys.
        Args:
            chrom_keys: chromophore keys to be fit on. If not none, chrom_keys are fit on instead of the wavelength columns.
        '''
        # nm_start_end_str = ['Wavelength_' + str(self.nm_start), 'Wavelength_' + str(self.nm_end)]
        dot_data = { # Name, column key, and filepath of the absorbance data
            'HbO2': ('HbO2 cm-1/M', '/Users/cameronmay/Documents/HSI/Absorbances/Hb_HbO2 Absorbance.csv'),
            'Hb': ('Hb cm-1/M', '/Users/cameronmay/Documents/HSI/Absorbances/Hb_HbO2 Absorbance.csv'),
            'H2O': ('H2O 1/cm', '/Users/cameronmay/Documents/HSI/Absorbances/Water Absorbance.csv'),
            'Pheomelanin': ('Pheomelanin cm-1/M', '/Users/cameronmay/Documents/HSI/Absorbances/Pheomelanin.csv'),
            'Eumelanin': ('Eumelanin cm-1/M', '/Users/cameronmay/Documents/HSI/Absorbances/Eumelanin Absorbance.csv'),
            'fat': ('fat', '/Users/cameronmay/Documents/HSI/Absorbances/Fat Absorbance.csv'),
            'L': ('L', '/Users/cameronmay/Documents/HSI/Absorbances/LM Absorbance.csv'),
            'M': ('M', '/Users/cameronmay/Documents/HSI/Absorbances/LM Absorbance.csv'),
            'S': ('S', '/Users/cameronmay/Documents/HSI/Absorbances/S Absorbance.csv')
        }
        if self.chrom_keys is not None:
            print('Performing dot product...')
            chrom_interps = pd.DataFrame([])
            ## Take the dot product of each chromophore, save output in df, and save each dot 
            for key in self.chrom_keys:
                chrom_interp = ChromDotSpectra.chrom_dot_spectra(self.df, [self.nm_start,self.nm_end], dot_data[key][0], key, dot_data[key][1],
                                   self.d65_and_cones, self.scrambled_chrom, normalized=False, plot=False)
                chrom_interps = pd.concat([chrom_interps, chrom_interp], axis=0)
            self.col_names = self.chrom_keys # replace column names 
        else:
            chrom_interps = None
        return chrom_interps

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

    def process_fold(self, fold_num, train_index, test_index, i, uniq_ids, data_frac, y_categories, y_col, col_names, scale, optimize, stochastic, random_state):
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
        # Rescramble the chromophores every fold if we are scrambling. Else, we can save the chromophores that were initialized at the beginning of the run() in process_data()
        if self.scrambled_chrom:
            chrom_interp_fold = self.dot_product() 
        else:
            chrom_interp_fold = self.chrom_interps

        ## Select data to train on
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

        ## Optimize and fit model
        if optimize: 
            svc_params = self.optimize_hyperparameters(X_train, y_train)
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

        expected = ['EdemaFalse', 'EdemaTrue']
        if not np.array_equal(np.unique(y_train), np.array(expected)):
            raise ValueError(f'Unexpected y_train categories or order: {np.unique(y_train)}. Expected: {expected} \n This changes interpertation of the SVM coefficients')

        if optimize:
            if svc_params['kernel'] == 'linear':
                coef_fold = pd.DataFrame(svm.coef_[0].reshape(1, -1), columns=col_names)
            elif svc_params['kernel'] == 'rbf':
                coef_fold = pd.DataFrame(np.zeros((1, len(col_names))), columns=col_names)
        else:
            coef_fold = pd.DataFrame(svm.coef_[0].reshape(1, -1), columns=col_names)

        coef_fold['Negative'] = np.unique(y_train)[0] # Add the negative class label
        coef_fold['Positive'] = np.unique(y_train)[1] # Add the positive class label

        test.loc[:, 'correct'] = y_test == y_pred
        test.loc[:, 'yPred'] = y_pred
        test.loc[:, 'yTest'] = y_test

        ## Process the confusion matricies 
        confusion_fold = pd.DataFrame([])
        uniq_test_ids = test['ID'].unique()
        for ID in uniq_test_ids:
            test_id = test[(test['ID'] == ID)] 
            if not test_id.empty:
                cm = confusion_matrix(test_id['yTest'], test_id['yPred'], labels=y_categories)
                if cm.size != 1:
                    TP, FN, FP, TN = cm.ravel()
                    Sensitivity = TP / (TP + FN)
                    Specificity = TN / (TN + FP)
                    confusion_fold = pd.concat([confusion_fold, pd.DataFrame([[fold_num, ID, TN, FP, FN, TP, Sensitivity, Specificity]], columns=['foldNum', 'ID', 'TN', 'FP', 'FN', 'TP', 'Sensitivity', 'Specificity'])])

        ## Save the predictions for each fold
        pred_loc_fold = test[['ID', 'yPred', 'FloatName', 'correct', y_col]].copy()
        pred_loc_fold['foldNum'] = fold_num

        print(f'Patch Acc Fold {fold_num} iteration {i}: {np.mean(test["correct"])}')

        ## Make the ID_acc dataframe for this fold 
        ID_acc_fold = test.groupby(['ID', y_col])['correct'].mean()
        ID_acc_fold = ID_acc_fold.to_frame(name=f'Iter {i} Fold {fold_num}')
        ID_acc_fold.reset_index(inplace=True)

        return confusion_fold, pred_loc_fold, ID_acc_fold, coef_fold, chrom_interp_fold

    def train_and_evaluate(self):
        '''
        Trains the model and evaluates it using cross-validation
        '''
        for i in range(self.iterations):
            print(f'\n\n-------Starting iteration: {i} -------\n')
            random_state = np.random.randint(0, 4294967295)

            # sample a subset of df and split into folds
            data_frac = self.df.sample(n=self.selected_num, random_state=random_state) 
            while  not np.array_equal(self.uniq_ids, np.sort(data_frac['ID'].unique())): # Resample if the fraction has zero examples for any class
                data_frac = self.df.sample(n=self.selected_num, random_state=random_state)
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=random_state)

            # process each fold in parallel
            fold_output = Parallel(n_jobs=self.n_jobs)( 
                delayed(self.process_fold)(fold_num, train_index, test_index, i, self.uniq_ids, data_frac, 
                                                   self.y_categories, self.y_col, self.col_names, self.scale, 
                                                   self.optimize, self.stochastic, random_state)
                for fold_num, (train_index, test_index) in enumerate(kf.split(self.uniq_ids))
            )

            # joblib's Parrallel function (used for parallelization of fold processing) requires concatenation of the outputs 
            confusion_fold, pred_loc_fold, ID_acc_fold, coef_fold, chrom_interp_fold = zip(*fold_output)
            confusion_fold = pd.concat(confusion_fold, ignore_index=True)
            pred_loc_fold = pd.concat(pred_loc_fold, ignore_index=True)
            ID_acc_fold = pd.concat(ID_acc_fold, ignore_index=True)
            coef_fold = pd.concat(coef_fold, ignore_index=True)
            
            if any(x is None for x in chrom_interp_fold):
                chrom_interp_fold = None
            else:
                chrom_interp_fold = pd.concat(chrom_interp_fold, ignore_index=True)
            self.confusions = pd.concat([self.confusions, confusion_fold])
            self.PredLocs = pd.concat([self.PredLocs, pred_loc_fold])
            self.coefs = pd.concat([self.coefs, coef_fold])
            self.chrom_interp = pd.concat([self.chrom_interp, chrom_interp_fold])
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
        iter_columns = [col for col in self.IDaccs.columns if col.startswith('Iter')] # Find the columns that are iterations
        iteration = self.IDaccs.loc[has_folder_has_avg, iter_columns]

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
        self.IDaccs['ID'] = self.IDaccs['ID'].fillna(0).astype(int) # fill NaN IDs with 0 so we can convert to int for cleaner labels
        self.IDaccs.insert(2, 'Labels', self.IDaccs['ID'].astype(int).astype(str) + ' ' + self.IDaccs[self.y_col].astype(str)) # find labels for bar chart
        self.IDaccs.loc['Column Average', 'Labels'] = 'Column Average' # add column average label
        
        # Make a dataframe with the CI95 by edemaTrue and edemaFalse
        Folder_acc = self.IDaccs.groupby(self.y_col)[['ID Avg','ID 0.5 Rounded Avg'] + iter_columns].mean() # find accuracy by foldername; add to IDaccs
        Folder_acc = Folder_acc.reset_index(inplace=False)
        Folder_acc['Labels'] = Folder_acc[self.y_col]
        CI95_folder = Folder_acc[iter_columns].apply(lambda row: PostProcessor.bootstrap_series(row), axis=1)
        Folder_acc.insert(2, '95%CI', CI95_folder) # add 95% CI to Folder_acc
        Folder_acc.insert(2, 'Yerr', np.abs(Folder_acc['95%CI'] - Folder_acc['ID Avg'])) # add error needed for bar chart
        Folder_acc.insert(2, 'Yerr lower', np.vstack(Folder_acc['Yerr'])[:,0])
        Folder_acc.insert(3, 'Yerr upper', np.vstack(Folder_acc['Yerr'])[:,1])
        Folder_acc.insert(2, 'Yerr round', np.abs(Folder_acc['95%CI'] - Folder_acc[f'ID {thresh} Rounded Avg'])) # add error needed for bar chart
        Folder_acc.insert(2, 'Yerr lower round', np.vstack(Folder_acc['Yerr round'])[:,0])
        Folder_acc.insert(3, 'Yerr upper round', np.vstack(Folder_acc['Yerr round'])[:,1])
        
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
        coef_plot = self.coefs.copy()
        coef_plot = coef_plot.drop(['Negative', 'Positive'], axis=1) # drop the negative and positive class labels
        coef_plot = coef_plot.apply(lambda col: PostProcessor.bootstrap_series(col), axis=0)
        filename = self.filepath.split("/")[-1]
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        plt.plot(abs(coef_plot.median(axis=0)), label='abs(median)')
        plt.plot(coef_plot.iloc[0, :], label='lower 95% CI')
        plt.plot(coef_plot.iloc[1, :], label='upper 95% CI')
        plt.xticks(rotation=90)
        plt.ylabel('Coefficient Value')
        plt.title(f'{self.data_config} data:{self.filepath.split("/")[-1]} \n iterations: {self.iterations} n={self.selected_num} fracs={self.fracs}')
        plt.legend()
        plt.tight_layout()

        # find the confusion matricies for each ID
        confusion_sum = self.confusions[['ID','TN','FP','FN','TP']].groupby('ID').sum() 
        confusion_average = self.confusions[['ID','foldNum','Sensitivity','Specificity']].groupby('ID').mean()
        confusion = pd.concat([confusion_sum, confusion_average], axis=1)

        ## Save all of the dataframes, figures, and this script into a new folder for each model run
        folder = self.save_folder / f'Run {self.run_num}'
        folder.mkdir(parents=True, exist_ok=True)

        # Save parameters as a .txt file
        parameters_file = folder / 'parameters.txt'
        with open(parameters_file, 'w') as f:
            json.dump(parameter_dict[run_num], f, indent=4)

        self.IDaccs.loc['Info', 'Labels'] = f'input filename: {filename}' # add metadata
        self.IDaccs.to_csv(folder / 'IDaccs.csv') # Save the accuracy as csv
        # PredLocs.to_csv(f'/Users/cameronmay/Downloads/{date}PredLocs_n={selectedNum}i={iterations}.csv.gz', compression='gzip', index=False) # Save predictions with locations as csv
        confusion.to_csv(folder / 'confusion.csv') # Saves each confusion matrix grouped by ID
        self.confusions.to_csv(folder / 'confusions.csv') # Saves each confusion matrix without grouping by ID
        leg_fig.savefig(folder / f'leg__pct={self.fracs}i={self.iterations} {filename}.pdf') # saves rounded accuracy
        patch_fig.savefig(folder / f'patch__acc={patch_acc}pct={self.fracs}.pdf') # saves patch accuracy
        coef_fig.savefig(folder / f'coef__acc={patch_acc}pct={self.fracs}.pdf') # saves SVM coefficients plot
        self.coefs.to_csv(folder / f'coefs__acc={patch_acc}pct={self.fracs}.csv', index=False) # saves coefficinets used to make coefFig as a csv
        if self.scrambled_chrom:
            self.chrom_interp.to_csv(folder / 'chrom_interps.csv') # Saves chromophores each fold 
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
    # Can run several different sets of parameters in sucession. parameters_dict is a dictionary or a list of dicionaries and each dictionary is a seperate model run
    parameter_dict = [
        {
        'fracs':0.005, # Fraction of examples to include
        'filepath':'/Users/cameronmay/Downloads/Jul_19_NOCR_R1R2R3_AllWLS.csv', # filepath of our dataset
        # 'fracs': 1, 'filepath': '/Users/cameronmay/Downloads/Jul_19_NOCR_R1R2R3_AllWLS_medians.csv', 
        'iterations':10, # number of iterations to run the model
        'n_jobs': -1, # 1 is for running normally, any number above 1 is for parallelization, and -1 takes all of the available CPUs
        'y_col':'Foldername', # what column of the data from filepath to fit on
        'scale':True, # if using StandardScaler() for the SVM
        'optimize':False, # if optimizing, C, kernel, and gamma 
        'stochastic':False, # if using stochastic gradient descent (better for larger amounts of data). Else use linear SVM 
        'nm_start':451.18, #451.18 # the minimum wavelength to include in model fits. Is rounded to nearest camera wavelength. 
        'nm_end': 954.83, #954.83 # the maximum wavelength to include in model fits. Is rounded to nearest camera wavelength 
        'data_config' : 'Round 1, 2, & 3: peripheral/edemafalse + controls', # which rounds and which disease group to fit on. Check data_configs for options
        'chrom_keys': None, # None if using the camera wavelengths, else using the dot product of the skin chromophore as the features to fit on. The keys of dot_data are the options: ['HbO2', 'Hb', 'H2O', 'Pheomelanin', 'Eumelanin', 'fat', 'L', 'M', 'S'].
        'd65_and_cones': False, # If scaling the data by the daylight axis (d65). Used in conjunction with ['L','M','S'] as chrom_keys. Also does not convert hyperspectral data to apparent absorbance, instead it keeps it in reflectance.
        'scrambled_chrom': False, # If scrambling the chromophores to be the same size but completely random
        'apparent_abs': True}, # If converting from reflectance to apparent absorbance A = log10(1/R)
    ]
        
    replace_chromophores = False # If True, will run the model with each chromophore in chrom_keys. If False, will run the model with the chrom_keys specified in parameter_dict

    if replace_chromophores:
        # To do Same run over all of the different chromophores. To use, set one parameter_dict, and this section will add one chromophore to each model run.
        if len(parameter_dict) > 1:
            raise ValueError('If replace_chromophores is True, parameter_dict should be a single dictionary, not a list of dictionaries.')
        chromophores = ['HbO2', 'Hb', 'H2O', 'Pheomelanin', 'Eumelanin', 'fat', 'L', 'M', 'S'] 
        parameter_dict = [parameter_dict.copy() for chromophore in chromophores] # make a copy of the parameters_dict so that we can run multiple models with different parameters
        for i, chromophore in enumerate(chromophores):
            parameter_dict[i]['chrom_keys'] = [chromophore]

    save_folder = Path(f'/Users/cameronmay/Downloads/SpectrumClassifier2 {str(datetime.now().strftime("%Y-%m-%d %H %M"))}')
    for run_num, parameters in enumerate(parameter_dict):
        classifier = SpectrumClassifier(run_num, save_folder, **parameters)
        classifier.run()
    plt.show()