### Perform the pearson correlation between chromophores and the absolute value of the median of the SVM weights.
# How to use: add the raw SVM weights from SpectrumClassifier2.py to svm_weights_csv_path and uncomment the chromophores you want to correlate in chromophore_datasets

from scipy.stats import pearsonr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.offline as pyo
import matplotlib.pyplot as plt
import pathlib as path
 
#CONFIGURATION
svm_weights_csv_path = path.PosixPath('/Users/maycaj/Downloads/SpectrumClassifier2 2025-06-26 21 55/Run 0/coefs__acc=93.8pct=1i=30 May28_CR_FullRound1and2AllWLs_medians.csv')
chromophore_datasets = { # Pick which chromophores to correlate with
    'Hb_diff': '/Users/maycaj/Documents/HSI/Absorbances/HbO2 Absorbance.csv',
    'Hb_deoxy': '/Users/maycaj/Documents/HSI/Absorbances/HbO2 Absorbance.csv',
    'Hb_oxy':'/Users/maycaj/Documents/HSI/Absorbances/HbO2 Absorbance.csv',
    'fat': '/Users/maycaj/Documents/HSI/Absorbances/Fat Absorbance.csv',
    'water': '/Users/maycaj/Documents/HSI/Absorbances/Water Absorbance.csv',
    # 'eumelanin': '/Users/maycaj/Documents/HSI/Absorbances/Eumelanin Absorbance.csv',
    # 'pheomelanin': '/Users/maycaj/Documents/HSI/Absorbances/Pheomelanin.csv'
}
## All wavelengths
# nmStart, nmEnd = 375, 1043
# Hemoglobins Region
nmStart, nmEnd = 375, 600
 
 
df_weights = pd.read_csv(svm_weights_csv_path)
# df_weights.columns = df_weights.columns.astype(float)
df_weights_T = df_weights.T
df_weights_T['median_weight'] = df_weights_T.median(axis=1)
w_median_abs = df_weights_T['median_weight'].abs()
w_zscore = (w_median_abs - w_median_abs.mean()) / w_median_abs.std()
df_weights_plot = pd.DataFrame({
    'wavelength': [float(num) for num in df_weights_T.index],
    'SVM_weight_zscore': w_zscore,
    'SVM_weight_abs': w_median_abs
})
df_weights_plot = df_weights_plot[(df_weights_plot['wavelength'] >= nmStart) & (df_weights_plot['wavelength'] <= nmEnd)].sort_values('wavelength')

output_files = []
 
for label, chromo_path in chromophore_datasets.items():
    df_chromo = pd.read_csv(chromo_path)
    df_chromo.rename(columns={'lambda nm': 'wavelength'}, inplace=True)
    df_chromo = df_chromo[(df_chromo['wavelength'] >= nmStart) & (df_chromo['wavelength'] <= nmEnd)].copy()
    df_chromo.sort_values('wavelength', inplace=True)
 
    if label == 'Hb_diff':
        df_chromo['signal'] = (df_chromo['HbO2 cm-1/M'] - df_chromo['Hb cm-1/M']).abs()
        chromo_name = '|HbO₂ - Hb|'
    elif label == 'Hb_deoxy':
        chromo_name = f'{label.capitalize()} Absorbance'
        df_chromo['signal'] = df_chromo['Hb cm-1/M']
    elif label == 'HbO2_oxy':
        chromo_name = f'{label.capitalize()} Absorbance'
        df_chromo['signal'] = df_chromo['HbO2 cm-1/M']
    else:
        chromo_name = f'{label.capitalize()} Absorbance'
        signal_col = df_chromo.columns[1]
        df_chromo['signal'] = df_chromo[signal_col]
 
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
        title=f'{chromo_name} vs. SVM Weights (Z) ({sim_text})',
        xaxis=dict(title='Wavelength (nm)'),
        yaxis=dict(title=chromo_name, side='left', showgrid=False),
        yaxis2=dict(title='SVM Weights (Z)', overlaying='y', side='right', showgrid=False),
        template='plotly_white', height=500
    )
    html_path = svm_weights_csv_path.parent / f'pearson_r={r:.2f}_{label}_vs_svm.html'
    pdf_path = f'/Users/maycaj/Documents/HSI/Pearson/pearson_r={r:.2f}_{label}_vs_svm.pdf'
    pyo.plot(fig, filename=str(html_path), auto_open=True)
 
    # # Matplotlib version
    # plt.figure(figsize=(6, 3))
    # fig_mt, ax1 = plt.subplots()
    # ax1.plot(interp_x, interp_y, color='tab:blue', linewidth=1.2)
    # ax1.set_xlabel('Wavelength (nm)')
    # ax1.set_ylabel(chromo_name, color='tab:blue')
    # ax1.tick_params(axis='y', labelcolor='tab:blue')
    # ax2 = ax1.twinx()
    # ax2.plot(interp_x, svm_y, color='tab:orange', linestyle='dashed', linewidth=1.2)
    # ax2.set_ylabel('SVM Weights (Z)', color='tab:orange')
    # ax2.tick_params(axis='y', labelcolor='tab:orange')
    # plt.title(f'{chromo_name} vs. SVM Weights\n{sim_text}', fontsize=10)
    # plt.tight_layout()
    # plt.savefig(pdf_path, format='pdf')
    # plt.close()
 
    output_files.append((label, html_path, pdf_path))

print('All done')