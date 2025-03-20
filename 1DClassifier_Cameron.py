import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from skimage.filters import threshold_otsu

# ---------------------------
# User Settings
# ---------------------------
csv_path = '/Users/maycaj/Documents/HSI_III/Mar10_Noarm.csv' #'/Users/maycaj/Documents/HSI_III/Mar4_1x1_ContinuumRemoved_WLFiltered_Recrop_Gender.csv' 
# wavelength_of_interest = 577.94  # single wavelength for 1D analysis
single_wl_col = 'HbO2' # 'Wavelength_577.94'

# Specify the binary variable column and its two labels.
binary_variable = "Gender"       # e.g., "Gender" "Foldername"
positive_label = "Male"          # mapped to 1
negative_label = "Female"        # mapped to 0

# Folder to save all HTML files
output_folder = '/Users/maycaj/Documents/HSI_III/1D Classifier'
os.makedirs(output_folder, exist_ok=True)
csv_name = csv_path.split('/')[-1].split('.')[0]

# ---------------------------
# Load and Prepare Data
# ---------------------------
print('Loading CSV...')
df = pd.read_csv(csv_path)
print('Done loading CSV!')

# # Add gender if Csv does not have it
# df['Gender'] = ''
# df.loc[df['ID'].isin([10,11,13,15,18,19,26,29,30,32,33,34,37,38,39]), 'Gender'] = 'Male'
# df.loc[df['ID'].isin([12,14,20,21,22,23,24,27,31,35,36,40]), 'Gender'] = 'Female'

# # Only keep valid IDs
# valid_ids = {4,11,12,15,18,20,22,23,26,34,36} # Cellulitis IDs
# df = df[df['ID'].isin(valid_ids)]

# Remove Dr. Pare
df = df[df['ID']!= 0]

# Filter dataset:
# For rows where the ID is in valid_ids, keep everything.
# For rows with IDs not in valid_ids, only keep if Foldername is 'EdemaFalse'.
# df = df[(df["ID"].isin(valid_ids)) | ((~df["ID"].isin(valid_ids)) & (df["Foldername"] == "EdemaFalse"))]
# df = df[df['Foldername'] == 'EdemaFalse'] # Select only healthy patients

def convolve_spectra(data, nmStartEnd, spectraName, outputName, absorbPath, showFig):
    '''
    Takes dataframe and convolves the spectra with Hemoglobin's response curve to see the total absorbance of hemoglobin in the skin.
    Args:
        data: pandas dataframe
        nmStartEnd: [startingWavelength, endingWavelength]
        spectraName: Name of column in the absorbance csv
        absorbPath: path to .csv with the absorbances
        showFig: boolean whether or not to plot
    '''
    absorbCSV = pd.read_csv(absorbPath) # Load the HbO2 spectra
    skinWaves = list(data.loc[:,nmStartEnd[0]:nmStartEnd[1]].columns)
    skinWaves = [float(wavelength.split('_')[1]) for wavelength in skinWaves] # Find wavelengths in skindata
    # for wavelength in skinWavelengths:
    #     if min(np.abs(absorbCSV['lambda nm'] - wavelength)) > 


    absorbWaves = []
    for wavelength in skinWaves: # match the data wavelengths with the wavelengths on the absorbance spectra
        absorbWaves.append(min(absorbCSV['lambda nm'], key=lambda x: np.abs(x-wavelength))) 
    matchedWave = pd.DataFrame({'AbsorbWaves': absorbWaves, 'SkinWaves': skinWaves})

    # remove the Skin Wave Values that are outside of the range of the absorbance values
    maxSkin = float('inf')
    minSkin = float('inf')
    for wavelength in skinWaves: 
        maxDiff = np.abs(max(matchedWave['AbsorbWaves'])-wavelength)
        if maxDiff < maxSkin:
            maxSkin = maxDiff
            maxWavelength = wavelength
        minDiff = np.abs(min(matchedWave['AbsorbWaves'])-wavelength)
        if minDiff < minSkin:
            minSkin = minDiff
            minWavelength = wavelength
    matchedWave = matchedWave[matchedWave['SkinWaves'] >= minWavelength]
    matchedWave = matchedWave[matchedWave['SkinWaves'] <= maxWavelength]

    # 
    pass

    # def matchWavelengths(wavelengths1, wavelengths2):
    #     '''
    #     matches wavelengths such that each pair is the smallest difference from each other, and each match is only made once
    #     Args:
    #         wavelengths1: list of wavelengths
    #         wavelengths2: list of wavelengths
    #     '''
    #     if (min(wavelengths1) < min(wavelengths2)) and (max(wavelengths2) < max(wavelengths1)): # check if wavelengths2 is contained within wavelengths1
    #         smallWavelengths = wavelengths2
    #         largeWavelengths = wavelengths1
    #     elif (min(wavelengths2) < min(wavelengths1)) and (max(wavelengths1) < max(wavelengths2)): # check if wavelengths1 is contained within wavelengths2
    #         smallWavelengths = wavelengths1
    #         largeWavelengths = wavelengths2
    #     else:
    #         raise ValueError('One of the wavelenghts is not contained within the other')
    #     for wavelength in smallWavelengths:



    # matchedDF = pd.DataFrame([], columns=['Matched Skin','Matched Absorb','Error'])
    # for skinWavelength in skinWavelengths:
    #     min_diff = float('inf')
    #     for absorbWavelength in absorbWavelengths:
    #         diff = np.abs(absorbWavelength-skinWavelength)
    #         if diff < min_diff:
    #             min_diff = diff
    #             matchedAbsorbs = absorbWavelength

        

    matchedWave = matchedWave.merge(absorbCSV, how='left', left_on='AbsorbWaves', right_on='lambda nm')

    matchedAbsorbs = matchedWave[spectraName].values
    matchedAbsorbs = matchedAbsorbs/np.max(matchedAbsorbs) # Normalize values
    # data_selected = data.loc[:,nmStartEnd[0]:nmStartEnd[1]]
    data_selected = data.loc[:,'Wavelength_' + f"{minWavelength:.2f}":'Wavelength_' + f"{maxWavelength:.2f}"]
    data_selected = data_selected.values
    data[outputName] = np.dot(data_selected, matchedAbsorbs)

    # Normalize output
    data[outputName] = data[outputName] - data[outputName].min()
    data[outputName] = data[outputName] / data[outputName].max()

    if showFig: # plot the absorbance spectra next 
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=skinWaves, y=matchedAbsorbs, name=outputName, line=dict(color='green'))) # plot HbO2
        for i in range(data.shape[0]): # plot all of skin spectra
            ID = str(data.iloc[i,data.columns.get_loc('ID')])
            HbO2 = str(np.round(data.iloc[i, data.columns.get_loc(outputName)],2))
            Foldername = str(data.iloc[i,data.columns.get_loc('Foldername')])
            name = 'ID:' + ID + outputName + HbO2 + 'Foldername:' + Foldername
            if Foldername == 'EdemaTrue':
                line=dict(color='red')
            elif Foldername == 'EdemaFalse':
                line=dict(color='blue')
            fig.add_trace(go.Scatter(x=skinWaves, y=data.iloc[i,data.columns.get_loc(nmStartEnd[0]):data.columns.get_loc(nmStartEnd[1])+1], name=name, line=line))
        fig.update_layout(title='Skin Spectra with HbO2 convolution as label and HbO2 spectra')
        fig.show()
    return matchedAbsorbs




#df = df[df['Foldername'] == "EdemaFalse"]

# Extract all wavelength columns (assumes names like "Wavelength_501.51", etc.)
wavelength_columns = [col for col in df.columns if col.startswith("Wavelength_")]
wavelengths = [float(col.split("_")[1]) for col in wavelength_columns]

# Convert dataframe to absorbance: Absorbance = log_10(1/reflectance)
df[wavelength_columns] = np.log10(df[wavelength_columns]**-1)
HbO2Path = '/Users/maycaj/Documents/HSI_III/Absorbances/HbO2 Absorbance.csv'
convolve_spectra(df, [wavelength_columns[0],wavelength_columns[-1]], 'HbO2 cm-1/M', 'HbO2', HbO2Path, showFig=False) # Find HbO2 for each pixel
convolve_spectra(df, [wavelength_columns[0],wavelength_columns[-1]], 'Hb cm-1/M', 'Hb', HbO2Path, showFig=False) # Find Hb for each pixel
pheoPath = '/Users/maycaj/Documents/HSI_III/Absorbances/Pheomelanin.csv'
convolve_spectra(df, [wavelength_columns[0],wavelength_columns[-1]], 'Pheomelanin cm-1/M','Pheomelanin', pheoPath, showFig=False)
euPath = '/Users/maycaj/Documents/HSI_III/Absorbances/Eumelanin Absorbance.csv'
convolve_spectra(df, [wavelength_columns[0],wavelength_columns[-1]], 'Eumelanin cm-1/M','Eumelanin', euPath, showFig=False)

predLocs = pd.concat([df['HbO2'], df['Hb'], df['X'], df['Y'], df['FloatName'], df['ID'], df['Pheomelanin'], df['Eumelanin']],axis=1)
predLocs['HbO2/Hb'] = df['HbO2'] / df['Hb'] # Find relative hemoglobin saturation ratio
# Remove outlier ratios - removed too much of some images 
# predLocs = predLocs[predLocs['HbO2/Hb'] > predLocs['HbO2/Hb'].quantile(0.01)]
# predLocs = predLocs[predLocs['HbO2/Hb'] < predLocs['HbO2/Hb'].quantile(0.99)]
# Normalize ratio within each image
uniqFloat = predLocs['FloatName'].unique()
for float in uniqFloat: 
    mask = predLocs['FloatName'] == float # predictions for one ID
    # Normalize ratios from 0 to 1
    temp_col = predLocs.loc[mask, 'HbO2/Hb']
    predLocs.loc[mask, 'HbO2/Hb'] = temp_col - min(temp_col)
    temp_col = predLocs.loc[mask, 'HbO2/Hb']
    predLocs.loc[mask, 'HbO2/Hb'] = temp_col / max(temp_col)
predLocs.to_csv(output_folder + '/HbpredLocs' + csv_name + '.csv') # Save predLocs as a csv
print(f"Saved predLocs to {output_folder + '/HbpredLocs' + csv_name + '.csv'}")
# ---------------------------
# Compute Per-Patient Mean & Median
# ---------------------------
patient_mean = df.groupby(["ID","Foldername",binary_variable])[wavelength_columns].mean().reset_index() # group wavelengths by ID, Foldername, and binary variable
patient_median = df.groupby(["ID","Foldername",binary_variable])[wavelength_columns].median().reset_index()
matchedHb_mean = convolve_spectra(patient_mean, [wavelength_columns[0],wavelength_columns[-1]],showFig=False) # find HbO2 for the mean spectra
matchedHb_median = convolve_spectra(patient_median, [wavelength_columns[0],wavelength_columns[-1]],showFig=False)

df.columns = df.columns.str.strip()
if binary_variable not in df.columns:
    raise KeyError(f"'{binary_variable}' column not found in the CSV file.")

if str(df[binary_variable].dtype) != 'float64':
    df[binary_variable] = df[binary_variable].str.strip()

# Overall reference curves for multi-wavelength plots:
overall_mean_true = patient_mean[patient_mean[binary_variable] == positive_label][wavelength_columns].mean(axis=0)
overall_mean_false = patient_mean[patient_mean[binary_variable] == negative_label][wavelength_columns].mean(axis=0)
overall_median_true = patient_median[patient_median[binary_variable] == positive_label][wavelength_columns].mean(axis=0)
overall_median_false = patient_median[patient_median[binary_variable] == negative_label][wavelength_columns].mean(axis=0)

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
    
    # Hb Curve
    fig.add_trace(go.Scatter(x=wavelengths, y=matchedHb_mean, mode='lines',
                             name=f"HB Spectrum",
                             line=dict(color='green', width=4)))

    fig.update_layout(title=title,
                      xaxis_title="Wavelength (nm)",
                      yaxis_title="Reflectance",
                      hovermode="closest")
    return fig

def create_interactive_plot_median(per_patient_df, overall_true, overall_false, title):
    fig = go.Figure()
    pos_data = per_patient_df[per_patient_df[binary_variable] == positive_label]
    for _, row in pos_data.iterrows():
        pid = row["ID"]
        yvals = [row[col] for col in wavelength_columns]
        fig.add_trace(go.Scatter(x=wavelengths, y=yvals, mode='lines',
                                 name=f"ID {pid} {positive_label}",
                                 line=dict(color="blue"), opacity=0.3))
    neg_data = per_patient_df[per_patient_df[binary_variable] == negative_label]
    for _, row in neg_data.iterrows():
        pid = row["ID"]
        yvals = [row[col] for col in wavelength_columns]
        fig.add_trace(go.Scatter(x=wavelengths, y=yvals, mode='lines',
                                 name=f"ID {pid} {negative_label}",
                                 line=dict(color="red"), opacity=0.3))
    fig.add_trace(go.Scatter(x=wavelengths, y=overall_true.tolist(), mode='lines',
                             name=f"Overall Median {positive_label}",
                             line=dict(color="blue", width=4)))
    fig.add_trace(go.Scatter(x=wavelengths, y=overall_false.tolist(), mode='lines',
                             name=f"Overall Median {negative_label}",
                             line=dict(color="red", width=4)))
    # Hb Curve
    fig.add_trace(go.Scatter(x=wavelengths, y=matchedHb_median, mode='lines',
                             name=f"HB Spectrum",
                             line=dict(color='green', width=4)))
    fig.update_layout(title=title,
                      xaxis_title="Wavelength (nm)",
                      yaxis_title="Reflectance",
                      hovermode="closest")
    return fig

# create a plot with each spectra divided by patient and foldername
fig_mean = create_interactive_plot(patient_mean, overall_mean_true, overall_mean_false,
                                  f"Mean Reflectance by Patient and {binary_variable}\n(Overall Means in Bold)")
fig_median = create_interactive_plot_median(patient_median, overall_median_true, overall_median_false,
                                             f"Median Reflectance by Patient and {binary_variable}\n(Overall Medians in Bold)")

# write each of the plots to the output folder
pio.write_html(fig_mean, os.path.join(output_folder, "Mean_Reflectance.html"), auto_open=True)
pio.write_html(fig_median, os.path.join(output_folder, "Median_Reflectance.html"), auto_open=True)
pass
# --------------------------------------------------
# PART 2: ORIGINAL 1D "NUMBER LINE" LOO PLOTS (by unique patient)
# --------------------------------------------------
# Identify the single wavelength column (e.g., "Wavelength_577.94" or "Wavelength_718.32")
# single_wl_col = None
# for col in wavelength_columns:
#     wl = float(col.split("_")[1])
#     if abs(wl - wavelength_of_interest) < 1e-5:
#         single_wl_col = col
#         break
# if single_wl_col is None:
#     raise ValueError(f"Could not find Wavelength_{wavelength_of_interest} in the DataFrame.")

# --- Helper functions for threshold estimation ---
def compute_confusion_with_inversion(reflectances, labels, boundary):
    # Option A: predict 1 if r >= boundary, else 0.
    tpA = fpA = tnA = fnA = 0
    for r, lab in zip(reflectances, labels):
        pred = 1 if r >= boundary else 0
        if pred == 1 and lab == 1:
            tpA += 1
        elif pred == 1 and lab == 0:
            fpA += 1
        elif pred == 0 and lab == 0:
            tnA += 1
        elif pred == 0 and lab == 1:
            fnA += 1
    totalA = tpA + tnA + fpA + fnA
    overallA = (tpA + tnA) / totalA if totalA else 0

    # Option B: inverted prediction: predict 0 if r >= boundary, else 1.
    tpB = fpB = tnB = fnB = 0
    for r, lab in zip(reflectances, labels):
        pred = 0 if r >= boundary else 1
        if pred == 1 and lab == 1:
            tpB += 1
        elif pred == 1 and lab == 0:
            fnB += 1
        elif pred == 0 and lab == 0:
            tnB += 1
        elif pred == 0 and lab == 1:
            fpB += 1
    totalB = tpB + tnB + fpB + fnB
    overallB = (tpB + tnB) / totalB if totalB else 0

    if overallA >= overallB:
        acc_true = tpA / (tpA + fnA) if (tpA+fnA) else 0
        acc_false = tnA / (tnA+fpA) if (tnA+fpA) else 0
        return overallA, acc_true, acc_false, False
    else:
        acc_true = tpB / (tpB+fnB) if (tpB+fnB) else 0
        acc_false = tnB / (tnB+fpB) if (tnB+fpB) else 0
        return overallB, acc_true, acc_false, True

def find_best_threshold_brute(reflectances, labels):
    unique_vals = sorted(set(reflectances))
    best_acc = 0.0
    best_thr = None
    best_acc_true = 0.0
    best_acc_false = 0.0
    best_invert = False
    for i in range(len(unique_vals)-1):
        thr = 0.5 * (unique_vals[i] + unique_vals[i+1])
        overall, acc_true, acc_false, invert = compute_confusion_with_inversion(reflectances, labels, thr)
        if overall > best_acc:
            best_acc = overall
            best_thr = thr
            best_acc_true = acc_true
            best_acc_false = acc_false
            best_invert = invert
    if unique_vals:
        for thr in [unique_vals[0]-1e-6, unique_vals[-1]+1e-6]:
            overall, acc_true, acc_false, invert = compute_confusion_with_inversion(reflectances, labels, thr)
            if overall > best_acc:
                best_acc = overall
                best_thr = thr
                best_acc_true = acc_true
                best_acc_false = acc_false
                best_invert = invert
    return best_thr, best_acc, best_acc_true, best_acc_false, best_invert

def find_threshold_youden(reflectances, labels):
    unique_vals = sorted(set(reflectances))
    best_j = -np.inf
    best_thr = None
    best_acc = 0.0
    best_acc_true = 0.0
    best_acc_false = 0.0
    best_invert = False
    for i in range(len(unique_vals)-1):
        thr = 0.5 * (unique_vals[i] + unique_vals[i+1])
        overall, acc_true, acc_false, invert = compute_confusion_with_inversion(reflectances, labels, thr)
        J = acc_true + acc_false - 1
        if J > best_j:
            best_j = J
            best_thr = thr
            best_acc = overall
            best_acc_true = acc_true
            best_acc_false = acc_false
            best_invert = invert
    if unique_vals:
        for thr in [unique_vals[0]-1e-6, unique_vals[-1]+1e-6]:
            overall, acc_true, acc_false, invert = compute_confusion_with_inversion(reflectances, labels, thr)
            J = acc_true + acc_false - 1
            if J > best_j:
                best_j = J
                best_thr = thr
                best_acc = overall
                best_acc_true = acc_true
                best_acc_false = acc_false
                best_invert = invert
    return best_thr, best_acc, best_acc_true, best_acc_false, best_invert

def find_threshold_otsu(reflectances, labels):
    try:
        thr = threshold_otsu(np.array(reflectances))
    except Exception:
        thr = None
    overall, acc_true, acc_false, invert = compute_confusion_with_inversion(reflectances, labels, thr)
    return thr, overall, acc_true, acc_false, invert

def fit_1d_svm_scaled(reflectances, labels):
    X = np.array(reflectances).reshape(-1, 1)
    y = np.array(labels)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = SVC(kernel='linear', C=1000.0)
    clf.fit(X_scaled, y)
    w0 = clf.intercept_[0]
    w1 = clf.coef_[0][0]
    boundary_scaled = -w0 / w1 if abs(w1) > 1e-12 else 999999.9
    boundary_original = scaler.inverse_transform([[boundary_scaled]])[0, 0]
    overall, acc_true, acc_false, invert = compute_confusion_with_inversion(reflectances, labels, boundary_original)
    return boundary_original, overall, acc_true, acc_false, invert

def loo_decision_boundary_multi(df_in, single_col):
    '''
    Find the decision boundary for several different methods 
    Args:
        df_in: grouped dataframe
        single_col: column we are using to fit
    '''
    unique_ids = df_in["ID"].unique()
    results = {
        "brute": {"thresholds": [], "tp": 0, "tn": 0, "fp": 0, "fn": 0},
        "youden": {"thresholds": [], "tp": 0, "tn": 0, "fp": 0, "fn": 0},
        "otsu": {"thresholds": [], "tp": 0, "tn": 0, "fp": 0, "fn": 0},
        "svm": {"thresholds": [], "tp": 0, "tn": 0, "fp": 0, "fn": 0}
    }
    for uid in unique_ids:
        train = df_in[df_in["ID"] != uid]
        test = df_in[df_in["ID"] == uid]
        train_refs = train[single_col].tolist() # training X
        train_labels = [1 if row[binary_variable] == positive_label else 0 for _, row in train.iterrows()]
        test_refs = test[single_col].tolist() # testing X
        test_labels = [1 if row[binary_variable] == positive_label else 0 for _, row in test.iterrows()]
        
        # Brute-force method:
        thr_b, _, _, _, invert_b = find_best_threshold_brute(train_refs, train_labels)
        results["brute"]["thresholds"].append(thr_b)
        for r, lab in zip(test_refs, test_labels):
            pred = (0 if r >= thr_b else 1) if invert_b else (1 if r >= thr_b else 0)
            if pred == 1 and lab == 1:
                results["brute"]["tp"] += 1
            elif pred == 1 and lab == 0:
                results["brute"]["fp"] += 1
            elif pred == 0 and lab == 0:
                results["brute"]["tn"] += 1
            elif pred == 0 and lab == 1:
                results["brute"]["fn"] += 1
        
        # Youden's method:
        thr_y, _, _, _, invert_y = find_threshold_youden(train_refs, train_labels)
        results["youden"]["thresholds"].append(thr_y)
        for r, lab in zip(test_refs, test_labels):
            pred = (0 if r >= thr_y else 1) if invert_y else (1 if r >= thr_y else 0)
            if pred == 1 and lab == 1:
                results["youden"]["tp"] += 1
            elif pred == 1 and lab == 0:
                results["youden"]["fp"] += 1
            elif pred == 0 and lab == 0:
                results["youden"]["tn"] += 1
            elif pred == 0 and lab == 1:
                results["youden"]["fn"] += 1
        
        # Otsu's method:
        thr_o, _, _, _, invert_o = find_threshold_otsu(train_refs, train_labels)
        results["otsu"]["thresholds"].append(thr_o)
        for r, lab in zip(test_refs, test_labels):
            pred = (0 if r >= thr_o else 1) if invert_o else (1 if r >= thr_o else 0)
            if pred == 1 and lab == 1:
                results["otsu"]["tp"] += 1
            elif pred == 1 and lab == 0:
                results["otsu"]["fp"] += 1
            elif pred == 0 and lab == 0:
                results["otsu"]["tn"] += 1
            elif pred == 0 and lab == 1:
                results["otsu"]["fn"] += 1
        
        # # SVM method:
        # thr_s, _, _, _, invert_s = fit_1d_svm_scaled(train_refs, train_labels)
        # results["svm"]["thresholds"].append(thr_s)
        # for r, lab in zip(test_refs, test_labels):
        #     pred = (0 if r >= thr_s else 1) if invert_s else (1 if r >= thr_s else 0)
        #     if pred == 1 and lab == 1:
        #         results["svm"]["tp"] += 1
        #     elif pred == 1 and lab == 0:
        #         results["svm"]["fp"] += 1
        #     elif pred == 0 and lab == 0:
        #         results["svm"]["tn"] += 1
        #     elif pred == 0 and lab == 1:
        #         results["svm"]["fn"] += 1

    final_results = {}
    for method in results:
        thr_list = results[method]["thresholds"]
        avg_thr = np.mean(thr_list) if thr_list else None
        tp = results[method]["tp"]
        tn = results[method]["tn"]
        fp = results[method]["fp"]
        fn = results[method]["fn"]
        total = tp + tn + fp + fn
        overall_acc = (tp + tn) / total if total > 0 else 0
        acc_true = tp / (tp + fn) if (tp + fn) > 0 else 0
        acc_false = tn / (tn + fp) if (tn + fp) > 0 else 0
        final_results[method] = (avg_thr, overall_acc, acc_true, acc_false)
    return final_results


def create_1d_number_line_plot_loo_multi(df_in, single_col, title):
    reflectances = []
    labels = []  # 1 for positive, 0 for negative
    offsets = []
    hover_texts = []
    for _, row in df_in.iterrows():
        cond = row[binary_variable]
        lab = 1 if cond == positive_label else 0
        ref_val = row[single_col]
        reflectances.append(ref_val)
        labels.append(lab)
        offsets.append(0.2 if lab == 1 else -0.2)
        pid = row["ID"]
        hover_texts.append(f"ID={pid}, {cond}, {ref_val:.4f}")
    loo_results = loo_decision_boundary_multi(df_in, single_col)
    fig = go.Figure()
    for i in range(len(reflectances)):
        color = "blue" if labels[i] == 1 else "red"
        fig.add_trace(go.Scatter(
            x=[reflectances[i]],
            y=[offsets[i]],
            mode='markers',
            marker=dict(color=color, size=8),
            hovertext=[hover_texts[i]],
            hoverinfo="text",
            showlegend=True,
            name = f"ID: {df_in['ID'].values[i]} {df_in[binary_variable].values[i]}" # each condition label
        ))
    method_colors = {"brute": "black", "youden": "orange", "otsu": "purple", "svm": "green"}
    subtitle_parts = []
    for method, (avg_thr, overall_acc, acc_true, acc_false) in loo_results.items():
        if avg_thr is not None:
            fig.add_shape(
                type="line",
                x0=avg_thr, y0=-1,
                x1=avg_thr, y1=1,
                line=dict(color=method_colors[method], width=2, dash="dash")
            )
            fig.add_annotation(
                x=avg_thr, y=0.8,
                text=f"{method.capitalize()}={avg_thr:.3f}",
                showarrow=True, arrowhead=2, ax=40, ay=0
            )
            subtitle_parts.append(f"{method.capitalize()} Acc: Overall={overall_acc:.3f}, True={acc_true:.3f}, False={acc_false:.3f}")
    subtitle = "<br><sup>" + "; ".join(subtitle_parts) + "</sup>"
    fig.update_layout(
        title=title + subtitle,
        xaxis_title=f"Reflectance at {single_col}",
        yaxis_title="",
        hovermode="closest",
        yaxis=dict(range=[-1, 1])
    )
    return fig

fig_mean_1d_loo = create_1d_number_line_plot_loo_multi(patient_mean[["ID", binary_variable, single_wl_col]].copy(),
                                                        single_wl_col,
                                                        title=f"LOO Decision Boundaries at {single_wl_col} (Mean Spectra)")
fig_median_1d_loo = create_1d_number_line_plot_loo_multi(patient_median[["ID", binary_variable, single_wl_col]].copy(),
                                                          single_wl_col,
                                                          title=f"LOO Decision Boundaries at {single_wl_col} (Median Spectra)")

pio.write_html(fig_mean_1d_loo, os.path.join(output_folder, "Mean_1D_LOO.html"), auto_open=True)
pio.write_html(fig_median_1d_loo, os.path.join(output_folder, "Median_1D_LOO.html"), auto_open=True)


def loo_decision_boundary_per_patient_pair(df_in, single_col):
    """
    For each unique patient ID, leave that patient out and compute a threshold on training data.
    Then, for each (ID, class) in the test set, compute classification accuracy.
    Returns a dictionary (per method) mapping (ID, class) to accuracy.
    """
    results = {"brute": {}, "youden": {}, "otsu": {}, "svm": {}}
    unique_ids = df_in["ID"].unique()
    for uid in unique_ids:
        train = df_in[df_in["ID"] != uid]
        test = df_in[df_in["ID"] == uid]
        train_refs = train[single_col].tolist()
        train_labels = [1 if row[binary_variable] == positive_label else 0 for _, row in train.iterrows()]
        for method in results.keys():
            if method == "brute":
                thr, _, _, _, invert = find_best_threshold_brute(train_refs, train_labels)
            elif method == "youden":
                thr, _, _, _, invert = find_threshold_youden(train_refs, train_labels)
            elif method == "otsu":
                thr, _, _, _, invert = find_threshold_otsu(train_refs, train_labels)
            elif method == "svm":
                thr, _, _, _, invert = fit_1d_svm_scaled(train_refs, train_labels)
            else:
                thr, _, _, _, invert = find_best_threshold_brute(train_refs, train_labels)
            # Group test data by binary class
            for cls, grp in test.groupby(binary_variable):
                test_refs = grp[single_col].tolist()
                test_labels = [1 if row[binary_variable] == positive_label else 0 for _, row in grp.iterrows()]
                correct = sum(1 for r, lab in zip(test_refs, test_labels)
                              if ((0 if r >= thr else 1) if invert else (1 if r >= thr else 0)) == lab)
                acc = correct / len(test_refs) if len(test_refs) > 0 else None
                results[method][(uid, cls)] = acc
    return results

def create_bar_chart_per_pair(per_pair_results, method_name, title):
    x_labels = []
    accuracies = []
    for (pid, cls), acc in per_pair_results.items():
        x_labels.append(f"{pid} {cls}")
        accuracies.append(acc * 100)  # percentage
    fig = go.Figure()
    colors = ["blue" if cls == positive_label else "red" for (pid, cls) in per_pair_results.keys()]
    fig.add_trace(go.Bar(x=x_labels, y=accuracies, marker_color=colors))
    fig.update_layout(title=f"{title}<br><sup>{method_name.capitalize()} Method</sup>",
                      xaxis_title="Patient ID and Class",
                      yaxis_title="Accuracy (%)",
                      yaxis=dict(range=[0, 105]))
    return fig

loo_results_per_pair_mean = loo_decision_boundary_per_patient_pair(patient_mean[["ID", binary_variable, single_wl_col]].copy(), single_wl_col)
loo_results_per_pair_median = loo_decision_boundary_per_patient_pair(patient_median[["ID", binary_variable, single_wl_col]].copy(), single_wl_col)

# for method in ["brute", "youden", "otsu", "svm"]:
#     fig_bar_mean = create_bar_chart_per_pair(loo_results_per_pair_mean[method], method,
#                                               f"LOO Accuracy per Patient-Class at {single_wl_col} nm (Mean Spectra)")
#     fig_bar_median = create_bar_chart_per_pair(loo_results_per_pair_median[method], method,
#                                                 f"LOO Accuracy per Patient-Class at {single_wl_col} nm (Median Spectra)")
#     pio.write_html(fig_bar_mean, os.path.join(output_folder, f"BarChart_Pair_Mean_{method.capitalize()}.html"), auto_open=True)
#     pio.write_html(fig_bar_median, os.path.join(output_folder, f"BarChart_Pair_Median_{method.capitalize()}.html"), auto_open=True)


def create_roc_curve(df_in, single_col, method):
    reflectances = df_in[single_col].tolist()
    labels = [1 if row[binary_variable] == positive_label else 0 for _, row in df_in.iterrows()]
    if method == "brute":
        thr, _, _, _, invert = find_best_threshold_brute(reflectances, labels)
    elif method == "youden":
        thr, _, _, _, invert = find_threshold_youden(reflectances, labels)
    elif method == "otsu":
        thr, _, _, _, invert = find_threshold_otsu(reflectances, labels)
    elif method == "svm":
        thr, _, _, _, invert = fit_1d_svm_scaled(reflectances, labels)
    else:
        thr, _, _, _, invert = find_best_threshold_brute(reflectances, labels)
    
    fprA, tprA, _ = roc_curve(labels, reflectances)
    aucA = auc(fprA, tprA)
    fprB, tprB, _ = roc_curve(labels, -np.array(reflectances))
    aucB = auc(fprB, tprB)
    
    if aucA >= aucB:
        best_fpr, best_tpr, best_auc = fprA, tprA, aucA
        chosen_thr = thr
        desc = "Direct"
    else:
        best_fpr, best_tpr, best_auc = fprB, tprB, aucB
        chosen_thr = thr
        desc = "Inverted"
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=best_fpr, y=best_tpr, mode='lines',
                             name=f"ROC curve (AUC = {best_auc:.3f})",
                             line=dict(color='darkorange', width=2)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                             name='Chance', line=dict(color='navy', width=1, dash='dash')))
    fig.update_layout(title=f"ROC Curve (Median Data) for {method.capitalize()} Method (Thr={chosen_thr:.3f}, {desc})",
                      xaxis_title="False Positive Rate",
                      yaxis_title="True Positive Rate",
                      hovermode="closest")
    return fig

# # Generate ROC curves for all methods using the median data
# for method in ["brute", "youden", "otsu", "svm"]:
#     roc_fig = create_roc_curve(patient_median[["ID", binary_variable, single_wl_col]].copy(), single_wl_col, method=method)
#     pio.write_html(roc_fig, os.path.join(output_folder, f"ROC_Median_{method.capitalize()}.html"), auto_open=True)

# Save PredLocs

print("All interactive multi‑wavelength, 1D LOO, bar chart, and ROC plots have been generated and saved to:")
print(output_folder)
print('All done :)')
pass