### Plots chromophores using the absorbance CSVs. Saves the plot as a PDF and HTML file and also saves script.
# How to use: Adjust chrom_keys to be the keys of dot_data which you want to plot

import plotly.graph_objects as go
import pandas as pd
import pathlib
import numpy as np
import shutil
import sys
import os

dot_data = { # Name, column key, and filepath of the absorbance data
    'HbO2': ('HbO2 cm-1/M', 'Hb_HbO2 Absorbance.csv'),
    'Hb': ('Hb cm-1/M', 'Hb_HbO2 Absorbance.csv'), 
    # 'H2O': ('H2O 1/cm', 'Water Absorbance.csv'),
    'Pheomelanin': ('Pheomelanin cm-1/M', 'Pheomelanin.csv'),
    'Eumelanin': ('Eumelanin cm-1/M', 'Eumelanin Absorbance.csv'),
    # 'fat': ('fat', 'Fat Absorbance.csv'),
    'L': ('L', 'LM Absorbance.csv'),
    'M': ('M', 'LM Absorbance.csv'),
    'S': ('S', 'S Absorbance.csv'),
    # 'D65': ('D65', 'CIE_std_illum_D65.csv'),
    }

chrom_keys = list(dot_data.keys()) # plotting all of the chromophores

nmStart = 400
nmEnd = 1000

absorb_folder = pathlib.Path('/Users/cameronmay/Documents/HSI/Absorbances')
output_folder = pathlib.Path("/Users/cameronmay/Downloads/ChromophoresPlots")
os.makedirs(output_folder, exist_ok=True)
if chrom_keys is not None:
    fig = go.Figure()
    for chrom_key in chrom_keys:
        absorbPath = absorb_folder / dot_data[chrom_key][1]
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
    fig.write_image(output_folder / "chromophores_plot.pdf")  # Save as PDF
    fig.write_html(output_folder / "chromophores_plot.html")  # Save as HTML
    fig.show()

shutil.copy(sys.argv[0], output_folder / 'PlotChromophores.py')
