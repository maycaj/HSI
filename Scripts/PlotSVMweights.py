import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import find_peaks

## Plot the relevant absorbances

def get_absorbances(absPath, col_name):
    '''
    get normalized absorbance values from csvs
    absPath: filepath of csv
    col_name: name of column with absorbance values
    '''
    absCSV = pd.read_csv(absPath)
    absCSV = absCSV[(absCSV['lambda nm'] > 450) & (absCSV['lambda nm'] < 958)]
    wavelength = absCSV['lambda nm']
    absorbance = absCSV[col_name]
    absorbance = absorbance / max(absorbance) # scale so the max value is at 1
    # reflectance = [10**-abs for abs in absorbance]

    return wavelength, absorbance

coefs = pd.read_csv('/Users/maycaj/Downloads/DebugData.csvcoefs_n=47i=1.csv')

# Load your absorbance data
plt.figure(1)
# Plot 1: Absorbance curves
# wavelength, absorbance = get_absorbances('/Users/maycaj/Documents/HSI_III/Absorbances/Eumelanin Absorbance.csv','Eumelanin cm-1/M')
# plt.plot(wavelength, absorbance, label='Eumelanin', color='brown')
# wavelength, absorbance = get_absorbances('/Users/maycaj/Documents/HSI_III/Absorbances/Fat Absorbance.csv','fat')
# plt.plot(wavelength, absorbance, label='fat', color='yellow')
HbO2wavelength, HbO2absorbance = get_absorbances('/Users/maycaj/Documents/HSI_III/Absorbances/HbO2 Absorbance.csv','HbO2 cm-1/M')
plt.plot(HbO2wavelength, HbO2absorbance, label='HbO2', color='red')
Hbwavelength, Hbabsorbance = get_absorbances('/Users/maycaj/Documents/HSI_III/Absorbances/HbO2 Absorbance.csv','Hb cm-1/M')
plt.plot(Hbwavelength, Hbabsorbance, label='Hb', color='blue')

# plt.plot([516,516],[0,1], color='black') # local maxima in difference between Hb and HbO2
# plt.plot([560,560],[0,1], color='black') # local maxima in difference between Hb and HbO2
# plt.plot([578,578],[0,1], color='black') # local maxima in difference between Hb and HbO2

diff = pd.DataFrame(np.abs(Hbabsorbance - HbO2absorbance), columns=['diff'])
# diff = diff.rank(pct=True)
# diff = diff.round(1)
diff = diff - min(diff.values)
diff = diff / max(diff.values)
HemoglobinDiff = pd.concat([HbO2wavelength, HbO2absorbance, Hbabsorbance, diff], axis=1)
plt.plot(Hbwavelength, diff, label='abs(HbO2 - Hb)', color='black')
plt.ylabel('Absorbance')
plt.title('Absorbance of Common Chemophores')
plt.legend()


# Plot 2: SVM coefficients as colorbar
wavelengths = coefs.columns.values.astype(float)
colorbar = coefs.median(axis=0).values.reshape(1, -1)
colorbar = np.abs(colorbar)

# Create wavelength "edges" for pcolormesh
# We assume midpoints between each wavelength for bin edges
edges = np.concatenate([
    [wavelengths[0] - (wavelengths[1] - wavelengths[0]) / 2],
    (wavelengths[:-1] + wavelengths[1:]) / 2,
    [wavelengths[-1] + (wavelengths[-1] - wavelengths[-2]) / 2]
])

# Create Y edges for the single row 
y_edges = [0,0.1]

plt.pcolormesh(edges, y_edges, colorbar, cmap='Greys', shading='flat')
plt.xlabel('Wavelength (nm)')
plt.xticks([400,500,600,700,800,900,1000]) 

plt.tight_layout()
plt.legend()
cbar = plt.colorbar()
cbar.set_label('Decoding Significance')
cbar.set_ticks([np.squeeze(np.min(colorbar, axis=1)), np.squeeze(np.max(colorbar, axis=1))])
cbar.set_ticklabels(['No significance','Highly significant'])


# skin data: more positive means more reflectance
# SVM coefficients: positive means Edema, negative means no edema
#     positive coefficient means higher reflectance at that wavelength is more likely to be edema
# Converting to absorbance molecules: positive coefficient means absorbance is less likely to be edema

plt.show()


print('All done ;)')