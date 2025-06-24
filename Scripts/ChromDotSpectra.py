### Perform dot product between skin spectra and spectra of skin chemophores
# How to use: This script is generally used for taking the dot product of a spectra with chromophores. Imported by SpectrumClassifier2.py for main use. To plot the chromophores directly, add their keys to chrom_keys and run this script directly.

import numpy as np
import pandas as pd
import pyarrow.csv as pv
import plotly.graph_objects as go

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

def chrom_dot_spectra(data, nmStartEnd, spectraName, outputName, absorbPath, d60=True, scrambled_chrom=False, normalized=True, plot=True):
    '''
    Takes dataframe and performs dot product with Hemoglobin's response curve to see the total absorbance of hemoglobin in the skin.
    Args:
        data: main pandas dataframe with wavelength and demographic data
        nmStartEnd: [startingWavelength, endingWavelength] in data
        spectraName: Name of column in the absorbance csv
        absorbPath: path to .csv with the absorbances
        d60 (bool): if combining the spectra with the daylight axis
        normalized (bool): if dividing the resultant output by its max such that the highest number is 1
        plot (bool): if plotting
    '''
    skin_waves = list(data.loc[:,nmStartEnd[0]:nmStartEnd[1]].columns) # find the wavelengths captured by camera

    # find the wavelengths where the skin wavelengths, absorbance wavelengths, and (if applicable) d60 wavelengths overlap
    x_abs = pd.read_csv(absorbPath)['lambda nm'].values
    dot_waves = [col for col in skin_waves if (col >= min(x_abs)) & (col <= max(x_abs))]
    d60_path = '/Users/maycaj/Documents/HSI/Absorbances/CIE_std_illum_D65.csv'
    if d60: # if d60, ensure that the wavelengths are within the range of d60
        x_abs_d60 = pd.read_csv(d60_path)['lambda nm'].values
        dot_waves = [col for col in dot_waves if (col >= min(x_abs_d60)) & (col <= max(x_abs_d60))]
    
    x_interp, y_interp_chrom, x_abs, y_abs = load_and_interpolate(absorbPath,'lambda nm',spectraName,dot_waves)

    if scrambled_chrom: # if chromophores need to be scrambled
        y_interp_chrom = np.random.rand(np.size(y_interp_chrom))

    if plot:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=x_interp, y=y_interp_chrom, name=f'Interp {spectraName}'))
        fig1.add_trace(go.Scatter(x=x_abs, y=y_abs, name=f'Original {spectraName}'))
        fig1.show()

    # find wavelengths that are in range for both the chromophores and the skin
    data_selected = data.loc[:,dot_waves].values
    if d60: # multiply by daylight axis
        x_interp, y_interp_d60, x_abs_d60, y_abs_d60 = load_and_interpolate(d60_path,'lambda nm','D65',dot_waves)
        if plot:
            fig1.add_trace(go.Scatter(x=x_interp, y=y_interp_d60, name=f'Interp d60'))
            fig1.add_trace(go.Scatter(x=x_abs_d60, y=y_abs_d60, name=f'Original d60'))
            fig1.show()
        data_selected = data_selected * y_interp_d60 
    data_selected = np.where(data_selected == 0, 1e-10, data_selected) # Ensure there is no divide by zero error in next step
    data_selected = np.log10(1 / data_selected) # Absorbance = log10(1 / reflectance)
    data[outputName] = np.dot(data_selected, y_interp_chrom)

    # Normalize output
    if normalized: 
        # data[outputName] = data[outputName] - data[outputName].min() + 0.01 # Doing this gets rid of a lot of the ability to segment the veins and also messed with decoding accuracy. It is also not realistic because there is still absorbance occuring even at the raw data's lowest point.
        data[outputName] = data[outputName] / data[outputName].max()

    return y_interp_chrom

if __name__ == '__main__':
    ### Plot the L, M, and S absorbances pre and post interpolation next to d65

    chrom_keys = ['L','M','S','D65']

    dot_data = { # Name, column key, and filepath of the absorbance data
    'HbO2': ('HbO2 cm-1/M', '/Users/maycaj/Documents/HSI/Absorbances/HbO2 Absorbance.csv'),
    'Hb': ('Hb cm-1/M', '/Users/maycaj/Documents/HSI/Absorbances/HbO2 Absorbance.csv'),
    'H2O': ('H2O 1/cm', '/Users/maycaj/Documents/HSI_III/Absorbances/Water Absorbance.csv'),
    'Pheomelanin': ('Pheomelanin cm-1/M', '/Users/maycaj/Documents/HSI_III/Absorbances/Pheomelanin.csv'),
    'Eumelanin': ('Eumelanin cm-1/M', '/Users/maycaj/Documents/HSI_III/Absorbances/Eumelanin Absorbance.csv'),
    'fat': ('fat', '/Users/maycaj/Documents/HSI_III/Absorbances/Fat Absorbance.csv'),
    'L': ('L', '/Users/maycaj/Documents/HSI/Absorbances/LM Absorbance.csv'),
    'M': ('M', '/Users/maycaj/Documents/HSI/Absorbances/LM Absorbance.csv'),
    'S': ('S', '/Users/maycaj/Documents/HSI/Absorbances/S Absorbance.csv'),
    'D65': ('D65', '/Users/maycaj/Documents/HSI/Absorbances/CIE_std_illum_D65.csv'),
    }
    if chrom_keys is not None:
        fig = go.Figure()
        for chrom_key in chrom_keys:
            absorbPath = dot_data[chrom_key][1]
            value_column = dot_data[chrom_key][0]
            x_interp = pd.read_csv(absorbPath)['lambda nm']
            x_interp, y_interp_chrom, x_abs, y_abs = load_and_interpolate(absorbPath,'lambda nm',value_column, x_interp)
            fig.add_trace(go.Scatter(x=x_abs, y=y_abs, name=f'Original: {chrom_key}'))
            fig.add_trace(go.Scatter(x=x_interp, y=y_interp_chrom, name=f'Interpolated: {chrom_key}'))
        fig.show()