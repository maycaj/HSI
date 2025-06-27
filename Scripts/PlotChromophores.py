### Plots chromophores using the absorbance CSVs
# How to use: Adjust chrom_keys to be the keys of dot_data which you want to plot

import plotly.graph_objects as go
import pandas as pd

dot_data = { # Name, column key, and filepath of the absorbance data
    'HbO2': ('HbO2 cm-1/M', '/Users/maycaj/Documents/HSI/Absorbances/HbO2 Absorbance.csv'),
    'Hb': ('Hb cm-1/M', '/Users/maycaj/Documents/HSI/Absorbances/HbO2 Absorbance.csv'),
    'H2O': ('H2O 1/cm', '/Users/maycaj/Documents/HSI/Absorbances/Water Absorbance.csv'),
    'Pheomelanin': ('Pheomelanin cm-1/M', '/Users/maycaj/Documents/HSI/Absorbances/Pheomelanin.csv'),
    'Eumelanin': ('Eumelanin cm-1/M', '/Users/maycaj/Documents/HSI/Absorbances/Eumelanin Absorbance.csv'),
    'fat': ('fat', '/Users/maycaj/Documents/HSI/Absorbances/Fat Absorbance.csv'),
    # 'L': ('L', '/Users/maycaj/Documents/HSI/Absorbances/LM Absorbance.csv'),
    # 'M': ('M', '/Users/maycaj/Documents/HSI/Absorbances/LM Absorbance.csv'),
    # 'S': ('S', '/Users/maycaj/Documents/HSI/Absorbances/S Absorbance.csv'),
    # 'D65': ('D65', '/Users/maycaj/Documents/HSI/Absorbances/CIE_std_illum_D65.csv'),
    }

chrom_keys = list(dot_data.keys()) # plotting all of the chromophores

nmStart = 400
nmEnd = 1000

if chrom_keys is not None:
    fig = go.Figure()
    for chrom_key in chrom_keys:
        absorbPath = dot_data[chrom_key][1]
        value_column = dot_data[chrom_key][0]
        absob_csv = pd.read_csv(absorbPath)
        x = absob_csv['lambda nm'].values
        mask = (x >= nmStart) & (x <= nmEnd)
        x = x[mask]
        y = absob_csv[value_column].values
        y = y[mask]
        y = y / np.max(y)
        fig.add_trace(go.Scatter(x=x, y=y, name=f'{chrom_key}', mode='lines'))
    fig.update_layout(title='Chromophores',
                    xaxis_title='Wavelength (nm)',
                    yaxis_title='Absorbance',
                    yaxis_type = 'log',
                    xaxis_showgrid=False,
                    yaxis_showgrid=False,
                    plot_bgcolor='white',   # Set plot area background to white
                    )
    fig.update_yaxes(
        tickvals=[1e-4, 1e-3, 1e-2, 1e-1, 1, 10])  # Only show ticks at powers of 10
    fig.write_image("/Users/maycaj/Downloads/chromophores_plot.pdf")  # Save as PDF
    fig.show()
