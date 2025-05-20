### Perform dot product between skin spectra and spectra of skin chemophores
from HSI_Functions import convolve_spectra
import pandas as pd

## Load in Data
csv_path = '/Users/maycaj/Documents/HSI_III/PatchCSVs/Mar25_NoCR_PostDeoxyCrop.csv'
output_folder = '/Users/maycaj/Documents/HSI_III/1D Classifier'
nmStart = ''
nmEnd = ''
print('Loading CSV...')
df = pd.read_csv(csv_path)
print('Done loading CSV!')
csv_name = csv_path.split('/')[-1].split('.')[0]

# Remove Dr. Pare
df = df[df['ID']!= 0]

## Convolve Hb
# HbO2Path = '/Users/maycaj/Documents/HSI_III/Absorbances/HbO2 Absorbance.csv'
# convolve_spectra(df, [nmStart,nmEnd], 'HbO2 cm-1/M', 'HbO2', HbO2Path, showFig=False) # Find HbO2 for each pixel
# convolve_spectra(df, [nmStart,nmEnd], 'Hb cm-1/M', 'Hb', HbO2Path, showFig=False) # Find Hb for each pixel

## Make and Save predLocs (Location and prediction of each pixel)
df['HbO2'] = df.loc[:,'Wavelength_416.24']
df['Hb'] = df.loc[:,'Wavelength_431.19']
predLocs = df[['X','Y','FloatName','ID', 'HbO2','Hb']]
predLocs['Hb/HbO2'] = df['Hb'] / df['HbO2'] # Find hemoglobin oxygen saturation ratio from the ratio of oxyhemoblobin to total hemoglobin: https://onlinelibrary.wiley.com/doi/full/10.1111/srt.12074
predLocs = predLocs.drop(['HbO2','Hb'],axis=1)
uniqFloat = predLocs['FloatName'].unique()

for float in uniqFloat: 
    # Normalize ratios from 0 to 1 within each image for each columnn
    for outputName in ['Hb/HbO2','Pheomelanin','H2O','fat']:
        mask = predLocs['FloatName'] == float # predictions for one ID
        temp_col = predLocs.loc[mask, outputName]
        predLocs.loc[mask, outputName] = temp_col - min(temp_col)
        temp_col = predLocs.loc[mask, outputName]
        predLocs.loc[mask, outputName] = temp_col / max(temp_col)
predLocs.to_csv(output_folder + '/HbpredLocs' + csv_name + '.csv') # Save predLocs as a csv
print(f"Saved predLocs to {output_folder + '/HbpredLocs' + csv_name + '.csv'}")
