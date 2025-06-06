### Perform Fully Constrained (only positive components and they must sum to 1) Linear Least Squares on Hyperspectral Data
### Answers the question: What is the linear combination of chromophores that yields a spectrum that is closest to our observed spectrum?

# venv created with terminal command: /Users/maycaj/homebrew/bin/python3.11 -m venv .venv

import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from joblib import Parallel, delayed
import time
import pyarrow.csv as pv
from wakepy import keep

class LeastSquaresProcessor:
    """
    Class to perform Fully Constrained Linear Least Squares on Hyperspectral Data.
    """

    def __init__(self, chrom_files, nm_start=516, nm_end=932.93):
        """
        Initialize the processor with chromophore files and wavelength range.

        Args:
            chrom_files: List of tuples containing file paths and column names for chromophores.
            nm_start: Start of wavelength range.
            nm_end: End of wavelength range.
        """
        self.chrom_files = chrom_files
        self.nm_start = nm_start
        self.nm_end = nm_end
        self.chrom_interp = pd.DataFrame([])

    def load_df(self, csv_path):
        '''
        Args:
            csv_path: path to csv
        Returns:
            df: loaded dataframe
            filename (str): name of file
        '''
        path_split = csv_path.split('.')
        filename = path_split[0]
        filename = filename.split('/')[-1]

        start_time = time.time()
        print('Loading dataframe...')
        table = pv.read_csv(csv_path)
        df = table.to_pandas()
        # df = pd.read_csv(csv_path, nrows=420)
        print(f'Done loading dataframe ({np.round(time.time()-start_time,1)}s)')
        return df, filename

    def load_and_interpolate(self, file_path, wavelength_column, value_column, x_interp):
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
        """
        csv_data = pd.read_csv(file_path)
        if not csv_data[wavelength_column].is_monotonic_increasing:
            csv_data = csv_data.sort_values(by=wavelength_column)
        x_abs = csv_data[wavelength_column].values
        y_abs = csv_data[value_column].values
        y_abs = y_abs / max(y_abs) # Scale from 0 to 1
        y_interp = np.interp(x_interp, x_abs, y_abs)
        # y_interp[(x_interp<x_abs[0])|(x_interp>x_abs[-1])] = 0 # values outside of chromophore absorbance data are set to 0
        y_interp = y_interp - min(y_interp)
        if max(y_interp) != 0:
            y_interp = y_interp / max(y_interp)
        return x_interp, y_interp, x_abs, y_abs

    def interpolate_chromophores(self, wave_cols_num, plot_interp=True):
        """
        Interpolate absorbance so it matches the camera's wavelengths.
        Plot Original Chromophore absorbance vs their interpolations.
        Save Interpolated Chromophores to dataframe.
        """
        fig1 = go.Figure()
        for chrom_file in self.chrom_files:
            x_interp, y_interp, x_abs, y_abs = self.load_and_interpolate(*chrom_file, wave_cols_num)
            self.chrom_interp[f'xInterp {chrom_file[2]}'] = x_interp
            self.chrom_interp[f'yInterp {chrom_file[2]}'] = y_interp
            fig1.add_trace(go.Scatter(x=x_interp, y=y_interp, name=f'Interp: {chrom_file[2]}'))
            fig1.add_trace(go.Scatter(x=x_abs, y=y_abs, name=f'Original: {chrom_file[2]}'))

        first_x_col = next((col for col in self.chrom_interp.columns if col.startswith('xInterp')), None)

        self.chrom_interp['xInterp brightness'] = self.chrom_interp[first_x_col]
        self.chrom_interp['yInterp brightness'] = 0.5 # Add a column for brightness

        self.chrom_interp['xInterp slope'] = self.chrom_interp[first_x_col]
        self.chrom_interp['yInterp slope'] = pd.DataFrame([i / len(wave_cols_num) for i in range(len(wave_cols_num))])

        fig1.update_layout(title='Interpolated Data with Additional Chromophores', xaxis_title='Wavelength', yaxis_title='Absorbance')
        if plot_interp:
            fig1.show()

    def chrom2observed(self, i, df, x_cols, y_cols, wave_cols_num, wave_cols, plot_recon=True):
        '''
        Takes known chromophore spectras and the skin absorbance and outputs the estimated percent composition of each chromophore
        Args:
            M: Numpy matrix [n_wavelengths x n_chromophores]
            A_obs: Observed Absorbance of the skin
            to_plot: plots chromophores and observed Absorbance
        '''
        print(f'chrom2observed: Processing row \t{i} of \t{df.shape[0]} \t{np.round(i*100/df.shape[0],2)}%')

        A_obs = df.iloc[i,:]
        label = A_obs['FloatName'] + ' ' + A_obs['Foldername']
        A_obs = A_obs[wave_cols].values

        M = self.chrom_interp[y_cols].values # [n_wavelengths x n_chromophores]
    
        A_obs = A_obs - min(A_obs) # Normalize A_obs from 0 to 1
        A_obs = A_obs / max(A_obs)
        A_obs_smo = np.convolve(A_obs, np.array([1/3,1/3,1/3]),mode='same') # Smooth with a 3-point mean
        A_obs_smo[0], A_obs_smo[-1] = A_obs[0], A_obs[-1] # Keep the same first and last points so we don't have artifacts

        n_chrom = M.shape[1] # Number of chromophores

        # Objective function: squared error. This is the function to minimize
        def objective(f):
            return np.linalg.norm(A_obs_smo - M @ f) ** 2

        # Initial guess: equal proportions
        x0 = np.ones(n_chrom) / n_chrom

        # Bounds: each value must be >= 0
        bounds = [(0, None) for _ in range(n_chrom-1)]
        bounds.append((None,None)) # For slope 

        # Minimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds) #, bounds=bounds, constraints=cons)

        # Output
        if result.success:
            fractions = result.x
            # print("Fractions:", fractions)
        else:
            print("Optimization failed:", result.message)
        sq_error = round(result.fun,5)

        A_recon = M @ fractions # reconstruct the absorbance from the dot product of the chromophores with the fractions

        fracPercent = fractions / sum(fractions) # find the % of each wavelength
        fracPercent = pd.DataFrame([fracPercent], columns=y_cols)
        fracPercent['label'] = label
        fracPercent['sq_error'] = sq_error
        fracPercent['X'] = df.iloc[i,df.columns.get_loc('X')]
        fracPercent['Y'] = df.iloc[i,df.columns.get_loc('Y')]
        fracPercent['FloatName'] = df.iloc[i,df.columns.get_loc('FloatName')]

        ## Plot the chromophores and the Observed absorbance
        if plot_recon:
            fig2 = go.Figure()
            for x_col, y_col in zip(x_cols, y_cols):
                chrom_percent = str(round(fracPercent[y_col].values[0]*100,1)) # find the fraction of the chromophore that is present
                fig2.add_trace(go.Scatter(x=self.chrom_interp[x_col], y=self.chrom_interp[y_col], name=f'{y_col} {chrom_percent}%',visible='legendonly'))
            fig2.add_trace(go.Scatter(x=wave_cols_num, y=A_obs, name=f"A_obs X={fracPercent['Y'].values}Y={fracPercent['X'].values}")) # X and Y are switched in the csv
            fig2.add_trace(go.Scatter(x=wave_cols_num, y=A_obs_smo, name='A_obs_smoothed'))
            fig2.add_trace(go.Scatter(x=wave_cols_num, y=A_recon, name=f'A_recon Sq_error={sq_error}'))
            fig2.update_layout(title='Chromophores '+label, xaxis_title ='wavelength', yaxis_title='Chromophore absorbances', font={'size': 17})
            fig2.show()
        return fracPercent

    def get_least_squares(self, df, plot_interp=True, plot_recon=True, n_jobs=-1):
        '''
        Get least squares fit to each chromophore for each row in the dataframe 
        Args:
            df: dataframe with the wavelength data [n_examples x n_columns(wavelengths+others)]
            plot_interp: boolean — plot or not plot interpolated chromophores, observed absorbance (A_obs), and reconstructed absorbance (A_recon)
            plot_recon: boolean — plot or not plot interpolated chromophores, observed absorbance (A_obs), and reconstructed absorbance (A_recon)
            n_jobs: number of parallel subprocesses. -1 means as many as possible 1 means only one subprocess which is good for debugging
        returns:
            fracPercents: dataframe with fractions of each chromophores [n_examples x n_chromophores]
            df: dataframe with added columns for the chromophore interpolation
        '''
        wave_cols = [col for col in df.columns if col.startswith('Wavelength')] # Select wavelength columns
        wave_cols_num = [float(col.split('Wavelength_')[1]) for col in wave_cols] # Convert to numerical
        wave_cols_num = np.array(wave_cols_num)
        wave_cols = np.array(wave_cols)
        mask = (self.nm_start <= wave_cols_num) & (wave_cols_num <= self.nm_end)
        wave_cols = wave_cols[mask]
        wave_cols_num = wave_cols_num[mask]

        self.interpolate_chromophores(wave_cols_num, plot_interp)

        df[wave_cols] = np.log10(df[wave_cols].values**-1) # convert to absorbance

        x_cols = [col for col in self.chrom_interp.columns if col.startswith('xInterp')]
        y_cols = [col for col in self.chrom_interp.columns if col.startswith('yInterp')]

        fracPercents = Parallel(n_jobs=n_jobs)(
                delayed(self.chrom2observed)(i, df, x_cols, y_cols, wave_cols_num, wave_cols, plot_recon=plot_recon)
                for i in range(df.shape[0])
            )
        
        fracPercents = pd.concat(fracPercents, ignore_index=True)
        fracPercents.reset_index(drop=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
        df = pd.concat([fracPercents, df],axis=1) # add fracPercents to original dataframe
        return df, fracPercents

if __name__ == '__main__':
    ## Load in Skin Data
    chrom_files = [
        ('/Users/maycaj/Documents/HSI/Absorbances/HbO2 Absorbance.csv', 'lambda nm', 'HbO2 cm-1/M'),
        ('/Users/maycaj/Documents/HSI/Absorbances/HbO2 Absorbance.csv', 'lambda nm', 'Hb cm-1/M'),
        ('/Users/maycaj/Documents/HSI/Absorbances/Eumelanin Absorbance.csv', 'lambda nm', 'Eumelanin cm-1/M'),
        ('/Users/maycaj/Documents/HSI/Absorbances/Fat Absorbance.csv', 'lambda nm', 'fat'), 
        ('/Users/maycaj/Documents/HSI/Absorbances/Pheomelanin.csv', 'lambda nm', 'Pheomelanin cm-1/M'),
        # ('/Users/maycaj/Documents/HSI/Absorbances/Water Absorbance.csv', 'lambda nm', 'H2O 1/cm')
    ]

    processor = LeastSquaresProcessor(chrom_files)

    with keep.running(on_fail='warn'): # keeps running when lid is shut
        # Your Python script code here
        # This will continue running while the script is active, even if the lid is closed
        filepath = '/Users/maycaj/Documents/HSI/PatchCSVs/May_29_NOCR_FullRound1and2AllWLs.csv'
        df, filename = processor.load_df(filepath)

        # Select a small subset of images for efficiency purposes
        # df = df[df['FloatName'].isin(['Edema 12 image 3 float','Edema 36 Image 3 float','Edema 19 Image 1 float'])]
        df = df.sample(n=6)

        start_ls = time.time()
        print('Starting least squares...')
        df, fracPercents = processor.get_least_squares(df, False, True, -1)
        print(f'Done with least squares: {np.round(time.time()-start_ls,1)}s')

        # Save the output to a csv file 
        # df.to_csv('/Users/maycaj/Downloads' + filename + '_LLS.csv', index=False)
        fracPercents.to_csv('/Users/maycaj/Downloads/' + 'LLS_516to600_' + filename +'.csv', index=False)

        print('All done ;)')
    breakpoint()
        