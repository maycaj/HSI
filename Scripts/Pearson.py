### Perform the pearson correlation between chromophores and the absolute value of the median of the SVM weights.
# How to use: add the raw SVM weights from SpectrumClassifier2.py to svm_weights_csv_path and uncomment the chromophores you want to correlate in chromophore_datasets

from scipy.stats import pearsonr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.offline as pyo
import matplotlib.pyplot as plt
import pathlib as path
import shutil
import sys

 
#CONFIGURATION
# Path to the SVM weights CSV file and chromophore datasets
svm_weights_csv_path = path.PosixPath('/Users/cameronmay/Downloads/SpectrumClassifier2 2025-07-25 14 36/Run 0/coefs__acc=84.0pct=1.csv')
dot_data = { # Name, column key, and filepath of the absorbance data
    'HbO2': ('HbO2 cm-1/M', 'Hb_HbO2 Absorbance.csv'),
    'Hb': ('Hb cm-1/M', 'Hb_HbO2 Absorbance.csv'), 
    # 'Hb_diff': ('Hb_diff', 'Hb_HbO2 Absorbance.csv'), 
    # 'H2O': ('H2O 1/cm', 'Water Absorbance.csv'), # is out of range below 667
    # 'Pheomelanin': ('Pheomelanin cm-1/M', 'Pheomelanin.csv'),
    # 'Eumelanin': ('Eumelanin cm-1/M', 'Eumelanin Absorbance.csv'),
    'fat': ('fat', 'Fat Absorbance.csv'),
    # 'L': ('L', 'LM Absorbance.csv'),
    # 'M': ('M', 'LM Absorbance.csv'),
    # 'S': ('S', 'S Absorbance.csv'),
    # 'D65': ('D65', 'CIE_std_illum_D65.csv'),
    }
absorb_folder = path.PosixPath('/Users/cameronmay/Documents/HSI/Absorbances')
absolute_value = False  # If True, use the absolute value of the SVM weights; if False, use the raw SVM weights
negate = False  # If True, negate the SVM weights before plotting

## Specify the wavelength range for the analysis
# All wavelengths
nmStart, nmEnd = 375, 1043
# Hemoglobins Region
# nmStart, nmEnd = 500, 600
# Fat Region
# nmStart, nmEnd = 880, 1043

 ## Load in SVM weights and Z-score them
df_weights = pd.read_csv(svm_weights_csv_path)
# df_weights.columns = df_weights.columns.astype(float)
df_weights = df_weights.drop(['Negative','Positive'], axis=1) # Remove the class labels for analysis 
df_weights_T = df_weights.T
df_weights_T['median_weight'] = df_weights_T.median(axis=1)
if absolute_value:
    median_weights = df_weights_T['median_weight'].abs()
else:
    median_weights = df_weights_T['median_weight']
if negate:
    median_weights = -median_weights

w_zscore = (median_weights - median_weights.mean()) / median_weights.std()
df_weights_plot = pd.DataFrame({
    'wavelength': [float(num) for num in df_weights_T.index],
    'SVM_weight_zscore': w_zscore,
    'SVM_weight_abs': median_weights
})
df_weights_plot = df_weights_plot[(df_weights_plot['wavelength'] >= nmStart) & (df_weights_plot['wavelength'] <= nmEnd)].sort_values('wavelength')

output_files = []
 
 # Loop over all of the data in dot_data and plot the chromophore absorbance, SVM weights, and the Pearson correlation
for key, (label, filename) in dot_data.items():
    filepath = absorb_folder / path.PosixPath(filename)
    df_chromo = pd.read_csv(filepath)
    df_chromo.rename(columns={'lambda nm': 'wavelength'}, inplace=True)
    df_chromo = df_chromo[(df_chromo['wavelength'] >= nmStart) & (df_chromo['wavelength'] <= nmEnd)].copy()
    df_chromo.sort_values('wavelength', inplace=True)
 
    # Plot additional relationships 
    if key == 'Hb_diff':
        df_chromo['signal'] = (df_chromo['HbO2 cm-1/M'] - df_chromo['Hb cm-1/M']).abs()
        chromo_name = '|HbO₂ - Hb|'
    else:
        chromo_name = f'{key.capitalize()} Absorbance'
        df_chromo['signal'] = df_chromo[label]
 
    # Normalize
    mean = df_chromo['signal'].mean()
    std = df_chromo['signal'].std()
    df_chromo['signal_zscore'] = (df_chromo['signal'] - mean) / std
 
    # Interpolation
    interp_x = df_weights_plot['wavelength'].values
    interp_y = np.interp(interp_x, df_chromo['wavelength'].values, df_chromo['signal_zscore'].values)
    svm_y = df_weights_plot['SVM_weight_zscore'].values
 
    # Pearson correlation
    r, p = pearsonr(interp_y, svm_y)
    sim_text = f'Pearson r = {r:.3f}, p = {p:.1e}'
 
    # Plotly plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=interp_x, y=interp_y, mode='lines', name=chromo_name, yaxis='y1'))
    fig.add_trace(go.Scatter(x=interp_x, y=svm_y, mode='lines', name='SVM Weights (Z)', yaxis='y2', line=dict(dash='dot')))
    fig.update_layout(
        title=f'{chromo_name} vs. SVM Weights (Z) ({sim_text}) Absolute Value: {absolute_value}',
        xaxis=dict(title='Wavelength (nm)'),
        yaxis=dict(title=chromo_name, side='left', showgrid=False),
        yaxis2=dict(title='SVM Weights (Z)', overlaying='y', side='right', showgrid=False),
        template='plotly_white', height=500
    )
    html_path = svm_weights_csv_path.parent / f'pearson_r={r:.2f}_{key}.html'
    pdf_path = f'/Users/maycaj/Documents/HSI/Pearson/pearson_r={r:.2f}_{key}.pdf'
    pyo.plot(fig, filename=str(html_path), auto_open=True)

    output_files.append((label, html_path, pdf_path))

shutil.copy(sys.argv[0], html_path.parent / 'PlotChromophores.py')
print('All done')