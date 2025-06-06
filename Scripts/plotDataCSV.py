import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


filepath = '/Users/maycaj/Documents/HSI_III/Apr_8_CR_FullRound1and2.csv' # '/Users/maycaj/Documents/HSI_III/PatchCSVs/Feb4_1x1_ContinuumRemoved_WLFiltered.csv' # '/Users/maycaj/Documents/HSI_III/DebugData.csv'
nmStart, nmEnd = 'Wavelength_451.18', 'Wavelength_954.83'
binary_variable = "Foldername"       # e.g., "Gender" "Foldername"
positive_label = "EdemaTrue"          # mapped to 1
negative_label = "EdemaFalse"        # mapped to 0

print('Loading CSV...')
df = pd.read_csv(filepath)
print('Done loading CSV')
df = df[(df['ID'].isin([11,12,15,18,20,22,23,26,34,36,45,59,61,70])) | (df['Foldername'] =='EdemaFalse')] # Round 2: Select only cellulitis IDs or IDs in EdemaFalse Folder
df = df[df['ID'] != 0] # Remove Dr. Pare

df = pd.concat([df.loc[:,nmStart:nmEnd], df['Foldername'], df['ID']], axis=1)
df = df.groupby(['Foldername','ID']).median()
df.reset_index(inplace=True)
wavelength_columns = [col for col in df.columns if col.startswith("Wavelength_")]
wavelengths = [float(col.split("_")[1]) for col in wavelength_columns]
dfEdema = df[df['Foldername'] == 'EdemaTrue'].loc[:,nmStart:nmEnd].median()
dfHealthy = df[df['Foldername'] == 'EdemaFalse'].loc[:,nmStart:nmEnd].median()


def create_interactive_plot(per_patient_df, overall_true, overall_false, title):
    fig = go.Figure()
    # Plot positive class curves in blue
    pos_data = per_patient_df[per_patient_df[binary_variable] == positive_label]
    for _, row in pos_data.iterrows():
        pid = row["ID"]
        yvals = [row[col] for col in wavelength_columns]
        fig.add_trace(go.Scatter(x=wavelengths, y=yvals, mode='lines',
                                 name=f"ID {pid} {positive_label}",
                                 line=dict(color="blue"), opacity=0.3))
    # Plot negative class curves in red
    neg_data = per_patient_df[per_patient_df[binary_variable] == negative_label]
    for _, row in neg_data.iterrows():
        pid = row["ID"]
        yvals = [row[col] for col in wavelength_columns]
        fig.add_trace(go.Scatter(x=wavelengths, y=yvals, mode='lines',
                                 name=f"ID {pid} {negative_label}",
                                 line=dict(color="red"), opacity=0.3))
    # Overall bold curves
    fig.add_trace(go.Scatter(x=wavelengths, y=overall_true.tolist(), mode='lines',
                             name=f"Overall Mean {positive_label}",
                             line=dict(color="blue", width=4)))
    fig.add_trace(go.Scatter(x=wavelengths, y=overall_false.tolist(), mode='lines',
                             name=f"Overall Mean {negative_label}",
                             line=dict(color="red", width=4)))
    
    # # Hb Curve
    # fig.add_trace(go.Scatter(x=wavelengths, y=matchedHb_mean, mode='lines',
    #                          name=f"HB Spectrum",
    #                          line=dict(color='green', width=4)))

    fig.update_layout(title=title,
                      xaxis_title="Wavelength (nm)",
                      yaxis_title="Reflectance",
                      hovermode="closest")
    return fig

fig = create_interactive_plot(df, dfEdema, dfHealthy, 'Round 1 and 2: Cellulitis Median Edematous vs Non-Edematous Spectra')
fig.show()


print('All done:)')