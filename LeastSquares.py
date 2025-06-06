#!/Users/maycaj/Documents/HSI_III/.venv/bin/python3

### Perform Fully Constrained (only positive components and they must sum to 1) Linear Least Squares on Hyperspectral Data
### Answers the question: What is the linear combination of chromophores that yields a spectrum that is closest to our observed spectrum?

import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import multiprocessing as mp
from joblib import Parallel, delayed
import time
import pyarrow.csv as pv
from wakepy import keep

def load_df(csv_path):
    '''
    Args:
        csv_path: path to csv
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
    return df

def load_and_interpolate(file_path, wavelength_column, value_column, xInterp):
    """
    Load a CSV file and interpolate the values based on the given wavelength column and value column.
    
    Parameters:
        file_path (str): Path to the CSV file.
        wavelength_column (str): Column name for wavelengths.
        value_column (str): Column name for the values to interpolate.
        cam_wave (array-like): Wavelengths to interpolate to.
    
    Returns:
        xInterp: wavelength values that correspond with yInterp
        yInterp: yAbs values interpolated to match with the cam_wave
        xAbs: x original absorbance values (wavelength in nm)
        yAbs: y original absorbance values 
    """
    csv_data = pd.read_csv(file_path)
    if not csv_data[wavelength_column].is_monotonic_increasing:
        csv_data = csv_data.sort_values(by=wavelength_column)
    xAbs = csv_data[wavelength_column].values
    yAbs = csv_data[value_column].values
    yAbs = yAbs / max(yAbs) # Scale from 0 to 1
    yInterp = np.interp(xInterp, xAbs, yAbs)
    # yInterp[(xInterp<xAbs[0])|(xInterp>xAbs[-1])] = 0 # values outside of chromophore absorbance data are set to 0
    yInterp = yInterp - min(yInterp)
    if max(yInterp) != 0:
        yInterp = yInterp / max(yInterp)
    return xInterp,yInterp,xAbs,yAbs

def chrom2observed(i, xCols, yCols, chromInterp, wave_cols_num, wave_cols, plot_recon=True):
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
    

    M = chromInterp[yCols].values # [n_wavelengths x n_chromophores]
   
    A_obs = A_obs - min(A_obs) # Normalize A_obs from 0 to 1
    A_obs = A_obs / max(A_obs)

    n_chrom = M.shape[1] # Number of chromophores

    # Objective function: squared error. This is the function to minimize
    def objective(f):
        return np.linalg.norm(A_obs - M @ f) ** 2

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
    fracPercent = pd.DataFrame([fracPercent], columns=yCols)
    fracPercent['label'] = label
    fracPercent['sq_error'] = sq_error
    fracPercent['X'] = df.iloc[i,df.columns.get_loc('X')]
    fracPercent['Y'] = df.iloc[i,df.columns.get_loc('Y')]
    fracPercent['FloatName'] = df.iloc[i,df.columns.get_loc('FloatName')]

    ## Plot the chromophores and the Observed absorbance
    if plot_recon:
        fig2 = go.Figure()
        for xCol,yCol in zip(xCols,yCols):
            chromPercent = str(round(fracPercent[yCol].values[0]*100,1)) # find the fraction of the chromophore that is present
            fig2.add_trace(go.Scatter(x=chromInterp[xCol], y=chromInterp[yCol], name=f'{yCol} {chromPercent}%'))
        fig2.add_trace(go.Scatter(x=wave_cols_num, y=A_obs, name='A_obs'))
        fig2.add_trace(go.Scatter(x=wave_cols_num, y=A_recon, name=f'A_recon Sq_error={sq_error}'))
        fig2.update_layout(title='Chromophores '+label, xaxis_title ='wavelength', yaxis_title='Chromophore absorbances', font={'size': 17})
        fig2.show()
    return fracPercent

def getLeastSquares(df, plot_interp = True, plot_recon = True, n_jobs=-1):
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
    nmStart = 516
    nmEnd = 600
    mask = (nmStart<wave_cols_num)&(wave_cols_num<nmEnd)
    wave_cols = wave_cols[mask]
    wave_cols_num = wave_cols_num[mask]

    # Define file paths and column names
    chrom_files = [
        ('/Users/maycaj/Documents/HSI/Absorbances/HbO2 Absorbance.csv', 'lambda nm', 'HbO2 cm-1/M'),
        ('/Users/maycaj/Documents/HSI/Absorbances/HbO2 Absorbance.csv', 'lambda nm', 'Hb cm-1/M'),
        ('/Users/maycaj/Documents/HSI/Absorbances/Eumelanin Absorbance.csv', 'lambda nm', 'Eumelanin cm-1/M'),
        ('/Users/maycaj/Documents/HSI/Absorbances/Fat Absorbance.csv', 'lambda nm', 'fat'), 
        ('/Users/maycaj/Documents/HSI/Absorbances/Pheomelanin.csv', 'lambda nm', 'Pheomelanin cm-1/M'),
        # ('/Users/maycaj/Documents/HSI/Absorbances/Water Absorbance.csv', 'lambda nm', 'H2O 1/cm')
    ]

    ## Interpolate absorbance so it matches the camera's wavelengths
    ## Plot Original Chromophore absorbance vs their interpolations
    ## Save Interpolated Chromophores to dataframe
    chromInterp = pd.DataFrame([])
    fig1 = go.Figure()
    for chrom_file in chrom_files:
        xInterp,yInterp,xAbs,yAbs = load_and_interpolate(*chrom_file, wave_cols_num)
        chromInterp[f'xInterp {chrom_file[2]}'] = xInterp
        chromInterp[f'yInterp {chrom_file[2]}'] = yInterp
        fig1.add_trace(go.Scatter(x=xInterp,y=yInterp,name=f'Interp: {chrom_file[2]}'))
        fig1.add_trace(go.Scatter(x=xAbs,y=yAbs,name=f'Original: {chrom_file[2]}'))

    firstXcol = next((col for col in chromInterp.columns if col.startswith('xInterp')), None)

    chromInterp['xInterp brightness'] = chromInterp[firstXcol]
    chromInterp['yInterp brightness'] = 0.5 # Add a column for brightness

    chromInterp['xInterp slope'] = chromInterp[firstXcol]
    chromInterp['yInterp slope'] = pd.DataFrame([i/len(wave_cols) for i in range(len(wave_cols))])

    fig1.update_layout(title='Interpolated Data with Additional Chromophores',xaxis_title='Wavelength',yaxis_title='Absorbance')
    if plot_interp:
        fig1.show()

    df[wave_cols] = np.log10(df[wave_cols].values**-1) # convert to absorbance

    xCols = [col for col in chromInterp.columns if col.startswith('xInterp')]
    yCols = [col for col in chromInterp.columns if col.startswith('yInterp')]

    fracPercents = Parallel(n_jobs=n_jobs)(
            delayed(chrom2observed)(i, xCols, yCols, chromInterp, wave_cols_num, wave_cols, plot_recon=plot_recon)
            for i in range(df.shape[0])
        )
    
    fracPercents = pd.concat(fracPercents, ignore_index=True)
    fracPercents.reset_index(drop=True, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = pd.concat([fracPercents, df],axis=1) # add fracPercents to original dataframe
    return df, fracPercents

if __name__ == '__main__':
    ## Load in Skin Data
    with keep.running(on_fail='warn'): # keeps running when lid is shut
        # Your Python script code here
        # This will continue running while the script is active, even if the lid is closed
        filename = '/Users/maycaj/Documents/HSI/PatchCSVs/Mar25_NoCR_PostDeoxyCrop.csv'
        df = load_df(filename)

        # Select a small subset of images for efficiency purposes
        df = df[df['FloatName'].isin(['Edema 12 image 3 float','Edema 36 Image 3 float','Edema 19 Image 1 float'])]

        start_ls = time.time()
        print('Starting least squares...')
        df, fracPercents = getLeastSquares(df, False, False, -1)
        print(f'Done with least squares: {np.round(time.time()-start_ls,1)}s')

        # Save the output to a csv file 
        # df.to_csv('/Users/maycaj/Downloads' + filename + '_LLS.csv', index=False)
        fracPercents.to_csv('/Users/maycaj/Downloads/' + 'LLS_516to600_' + filename +'.csv', index=False)

        print('All done ;)')
    