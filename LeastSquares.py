### Perform Fully Constrained (only positive components and they must sum to 1) Linear Least Squares on Hyperspectral Data

import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go


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
    yInterp = yInterp / max(yInterp)
    return xInterp,yInterp,xAbs,yAbs

def chrom2observed(A_obs: np.ndarray, chromInterp, wave_cols_num, label, plot_recon=True):
    '''
    Takes known chromophore spectras and the skin absorbance and outputs the estimated percent composition of each chromophore
    Args:
        M: Numpy matrix [n_wavelengths x n_chromophores]
        A_obs: Observed Absorbance of the skin
        to_plot: plots chromophores and observed Absorbance
    '''
    xCols = [col for col in chromInterp.columns if col.startswith('xInterp')]
    chromInterp['xInterp brightness'] = chromInterp[xCols[0]]
    chromInterp['yInterp brightness'] = 1 # Add a column for brightness

    xCols = [col for col in chromInterp.columns if col.startswith('xInterp')] # re-update x columns after adding brightness as a column
    yCols = [col for col in chromInterp.columns if col.startswith('yInterp')]

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
    bounds = [(0, None) for _ in range(n_chrom)]

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
    fracPercent = pd.DataFrame([fracPercent], columns=yCols) * 100
    fracPercent['label'] = label
    fracPercent['sq_error'] = sq_error

    ## Plot the chromophores and the Observed absorbance
    if plot_recon:
        fig2 = go.Figure()
        for xCol,yCol in zip(xCols,yCols):
            chromPercent = str(round(fracPercent[yCol].values[0],1)) # find the fraction of the chromophore that is present
            fig2.add_trace(go.Scatter(x=chromInterp[xCol], y=chromInterp[yCol], name=f'{yCol} {chromPercent}%'))
        fig2.add_trace(go.Scatter(x=wave_cols_num, y=A_obs, name='A_obs'))
        fig2.add_trace(go.Scatter(x=wave_cols_num, y=A_recon, name=f'A_recon Sq_error={sq_error}'))
        fig2.update_layout(title='Chromophores '+label, xaxis_title ='wavelength', yaxis_title='Chromophore absorbances', font={'size': 17})
        fig2.show()
    return fracPercent

def getLeastSquares(df, plot_interp = True, plot_recon = True):
    '''
    Get least squares fit to each chromophore for each row in the dataframe 
    Args:
        df: dataframe with the wavelength data [n_examples x n_columns(wavelengths+others)]
        plot_interp: boolean — plot or not plot interpolated chromophores, observed absorbance (A_obs), and reconstructed absorbance (A_recon)
        plot_recon: boolean — plot or not plot interpolated chromophores, observed absorbance (A_obs), and reconstructed absorbance (A_recon)
    returns:
        fracPercents: dataframe with fractions of each chromophores [n_examples x n_chromophores]
    '''

    wave_cols = [col for col in df.columns if col.startswith('Wavelength')] # Select wavelength columns
    wave_cols_num = [float(col.split('Wavelength_')[1]) for col in wave_cols] # Convert to numerical

    # Define file paths and column names
    chrom_files = [
        ('/Users/maycaj/Documents/HSI_III/Absorbances/HbO2 Absorbance.csv', 'lambda nm', 'HbO2 cm-1/M'),
        ('/Users/maycaj/Documents/HSI_III/Absorbances/HbO2 Absorbance.csv', 'lambda nm', 'Hb cm-1/M'),
        ('/Users/maycaj/Documents/HSI_III/Absorbances/Eumelanin Absorbance.csv', 'lambda nm', 'Eumelanin cm-1/M'),
        ('/Users/maycaj/Documents/HSI_III/Absorbances/Fat Absorbance.csv', 'lambda nm', 'fat'),
        ('/Users/maycaj/Documents/HSI_III/Absorbances/Pheomelanin.csv', 'lambda nm', 'Pheomelanin cm-1/M'),
        ('/Users/maycaj/Documents/HSI_III/Absorbances/Water Absorbance.csv', 'lambda nm', 'H2O 1/cm')
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
    fig1.update_layout(title='Interpolated Data with Additional Chromophores',xaxis_title='Wavelength',yaxis_title='Absorbance')
    if plot_interp:
        fig1.show()

    df[wave_cols] = np.log10(df[wave_cols].values**-1) # convert to absorbance
    fracPercents = pd.DataFrame([])
    for i in range(df.shape[0]):
        # A_obs: (n_wavelengths,) observed mixed spectrum
        A_obs = df.iloc[i,:]
        label = A_obs['FloatName'] + ' ' + A_obs['Foldername']
        A_obs = A_obs[wave_cols].values
        fracPercent = chrom2observed(A_obs, chromInterp, wave_cols_num, label, plot_recon=plot_recon)
        fracPercents = pd.concat([fracPercent,fracPercents],axis=0)
    return fracPercents

## Load in Skin Data
csv_path = '/Users/maycaj/Documents/HSI_III/PatchCSVs/Mar25_NoCR_PostDeoxyCrop_Medians.csv' # Data with medians for each float only
filename = csv_path.split('/')[-1]
print('Loading CSV...')
df = pd.read_csv(csv_path)
print('Done loading CSV!')

fracPercents = getLeastSquares(df, False, False)

## Plot HbO2/Hb for edemaTrue vs edemaFalse
fracPercents['HbO2/Hb'] = fracPercents['yInterp HbO2 cm-1/M']/fracPercents['yInterp Hb cm-1/M']
edemaTrue = fracPercents[fracPercents['label'].str.contains('EdemaTrue')]
edemaFalse = fracPercents[fracPercents['label'].str.contains('EdemaFalse')]
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=edemaFalse['HbO2/Hb'],y=[0 for i in range(edemaFalse['HbO2/Hb'].shape[0])], mode='markers', name='edemaFalse'))
fig3.add_trace(go.Scatter(x=edemaTrue['HbO2/Hb'],y=[1 for i in range(edemaTrue['HbO2/Hb'].shape[0])], mode='markers', name='edemaTrue'))
fig3.update_layout(title=f'')
fig3.show()

print('All done ;)')