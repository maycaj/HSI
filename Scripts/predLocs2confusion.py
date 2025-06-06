## Creates a confusion matrix from the PredLocs csv that SpectrumClassifier2.py outputs

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

## Load in data
filepath = '/Users/maycaj/Downloads/Feb4confusion_n=28936i=1.csv'
df = pd.read_csv(filepath)
cm = df.sum(axis=0)
cm = cm.drop(['ID','foldNum'])

## Plot confusion matrix
TP = cm['TP']
TN = cm['TN']
FP = cm['FP']
FN = cm['FN']
cm_arr = np.array([[TP,FN],
                  [FP,TN]])
plt.figure()
sns.heatmap(cm_arr, fmt='.0f', annot=True, xticklabels=['True','False'],yticklabels=['True','False'], cmap='grey')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(filepath.split('/')[-1])
plt.show()

print('All done ;)')
