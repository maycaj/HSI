### Perform dot product between skin spectra and spectra of skin chemophores
import numpy as np
import pandas as pd
import pyarrow.csv as pv
import plotly.graph_objects as go

def chrom_dot_spectra(data, nmStartEnd, spectraName, outputName, absorbPath, normalized=True, plot=False):
    '''
    Takes dataframe and performs dot product with Hemoglobin's response curve to see the total absorbance of hemoglobin in the skin.
    Args:
        data: main pandas dataframe with wavelength and demographic data
        nmStartEnd: [startingWavelength, endingWavelength] in data
        spectraName: Name of column in the absorbance csv
        absorbPath: path to .csv with the absorbances
        showFig: boolean whether or not to plot
    '''
    # absorbCSV = pd.read_csv(absorbPath) # Load the HbO2 spectra
    skinWaves = list(data.loc[:,nmStartEnd[0]:nmStartEnd[1]].columns)
    skinWaves_float = [float(wavelength.split('_')[1]) for wavelength in skinWaves] # Find wavelengths in skindata

    # absorbWaves = []
    # for wavelength in skinWaves: # match the data wavelengths with the wavelengths on the absorbance spectra
    #     absorbWaves.append(min(absorbCSV['lambda nm'], key=lambda x: np.abs(x-wavelength))) 
    # matchedWave = pd.DataFrame({'AbsorbWaves': absorbWaves, 'SkinWaves': skinWaves})

    # # remove the Skin Wave Values that are outside of the range of the absorbance values
    # maxSkin = float('inf')
    # minSkin = float('inf')
    # for wavelength in skinWaves: # find the maximum and minimum wavelengths in the skin waves
    #     # Looking for the closest absorbance to the highest skin wavelength
    #     maxDiff = np.abs(max(matchedWave['AbsorbWaves'])-wavelength) # absolute difference between highest absorbance wavelength and each skin wavelength
    #     if maxDiff < maxSkin: # if the absolute difference is lower than all the others we have seen before, update the threshold for the absolute difference, and find the new max wavelength
    #         maxSkin = maxDiff
    #         maxWavelength = wavelength
    #     # Looking for the closest absorbance to the lowest skin wavelength
    #     minDiff = np.abs(min(matchedWave['AbsorbWaves'])-wavelength)
    #     if minDiff < minSkin:
    #         minSkin = minDiff
    #         minWavelength = wavelength

    # matchedWave = matchedWave[matchedWave['SkinWaves'] >= minWavelength]
    # matchedWave = matchedWave[matchedWave['SkinWaves'] <= maxWavelength]

    # matchedWave = matchedWave.merge(absorbCSV, how='left', left_on='AbsorbWaves', right_on='lambda nm')

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
    
    x_interp, y_interp_chrom, x_abs, y_abs= load_and_interpolate(absorbPath,'lambda nm',spectraName,skinWaves_float)
    if plot:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=x_interp, y=y_interp_chrom, name=f'Interp'))
        fig1.add_trace(go.Scatter(x=x_abs, y=y_abs, name=f'Original'))
        fig1.show()

    # matchedAbsorbs = matchedWave[spectraName].values
    # matchedAbsorbs = matchedAbsorbs/np.max(matchedAbsorbs) # Normalize absorbance values from 0 to 1
    # data_selected = data.loc[:,'Wavelength_' + f"{minWavelength:.2f}":'Wavelength_' + f"{maxWavelength:.2f}"]
    data_selected = data[skinWaves]
    data_selected = np.log10(data_selected.values**-1) # Absorbance = log10(1 / reflectance)
    data[outputName] = np.dot(data_selected, y_interp_chrom)

    # Normalize output
    if normalized: 
        data[outputName] = data[outputName] - data[outputName].min() + 0.01
        data[outputName] = data[outputName] / data[outputName].max()

    return y_interp_chrom

if __name__ == '__main__':
    import pandas as pd
    import plotly.graph_objects as go

    ## Load in Data
    csv_path = '/Users/maycaj/Documents/HSI/PatchCSVs/May_29_NOCR_FullRound1and2AllWLs.csv'
    filename = csv_path.split('/')[-1]
    output_folder = '/Users/maycaj/Downloads/'
    nmStart = 'Wavelength_411.27' #'Wavelength_451.18'
    nmEnd = 'Wavelength_1004.39' #'Wavelength_954.83'
    print('Loading CSV...')
    table = pv.read_csv(csv_path)
    df = table.to_pandas()
    print('Done loading CSV!')
    csv_name = csv_path.split('/')[-1].split('.')[0]

    # Select only Patients of interest
    # df = df[df['ID']!= 0] # Remove Dr. Pare
    # df = df[df['ID'] <= 40] # Select only round 1
    # df = df[df['FloatName'].isin(['Edema 12 image 3 float','Edema 36 Image 3 float','Edema 19 Image 1 float'])]

    ## Convolve Hb
    HbO2Path = '/Users/maycaj/Documents/HSI/Absorbances/HbO2 Absorbance.csv'
    chrom_dot_spectra(df, [nmStart,nmEnd], 'HbO2 cm-1/M', 'HbO2', HbO2Path) # Find HbO2 for each pixel
    chrom_dot_spectra(df, [nmStart,nmEnd], 'Hb cm-1/M', 'Hb', HbO2Path) # Find Hb for each pixel

    # Make and Save predLocs (Location and prediction of each pixel)
    predLocs = df[['X','Y','FloatName','ID', 'HbO2','Hb']]
    predLocs['HbO2/Hb'] = df['HbO2'] / df['Hb'] # Find hemoglobin oxygen saturation ratio from the ratio of oxyhemoblobin to total hemoglobin: https://onlinelibrary.wiley.com/doi/full/10.1111/srt.12074
    predLocs = predLocs.drop(['HbO2','Hb'],axis=1)
    uniqFloat = predLocs['FloatName'].unique()

    for float in uniqFloat: 
        # Normalize ratios from 0 to 1 within each image for each columnn
        for outputName in ['HbO2/Hb']:
            mask = predLocs['FloatName'] == float # predictions for one ID
            temp_col = predLocs.loc[mask, outputName]
            predLocs.loc[mask, outputName] = temp_col - min(temp_col)
            temp_col = predLocs.loc[mask, outputName]
            predLocs.loc[mask, outputName] = temp_col / max(temp_col)
        
    predLocs.to_csv(output_folder + 'PredLocs_' + csv_name + '.csv') # Save predLocs as a csv
    print(f"Saved predLocs to {output_folder + '/HbpredLocs' + csv_name + '.csv'}")
    # df['HbO2/Hb'] = df['HbO2'] / df['Hb']
    # edemaTrue = df[df['Foldername'] == 'EdemaTrue']
    # edemaFalse = df[df['Foldername'] == 'EdemaFalse']
    # fig3 = go.Figure()
    # fig3.add_trace(go.Scatter(x=edemaFalse['HbO2/Hb'],y=[0 for i in range(edemaFalse['HbO2/Hb'].shape[0])], mode='markers', name='edemaFalse'))
    # fig3.add_trace(go.Scatter(x=edemaTrue['HbO2/Hb'],y=[1 for i in range(edemaTrue['HbO2/Hb'].shape[0])], mode='markers', name='edemaTrue'))
    # fig3.update_layout(title=f'{filename} edemaTrue vs edemaFalse dot product HbO2/Hb', xaxis_title='HbO2/Hb')
    # fig3.show()
    breakpoint()
