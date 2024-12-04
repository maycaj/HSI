import scipy.io
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
import itertools
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation
import re

# initialize filepath & number of samples
filepath = '/Users/maycaj/Documents/HSI_III/spectrums-12-1-24_EdemaTF_imgNum_5.csv' # '/Users/maycaj/Documents/HSI_III/spectrums-12-1-24_EdemaTF_imgNum.csv'
selectedNums = [3000] # max is 49652 for 11x11 patches
iterations = 300

X = np.genfromtxt(filepath, delimiter=',') #read in the .csv as a npy variable
X = X[1:,1:] # get rid of y column and column labels
y = pd.read_csv(filepath) # pull y column
y = y.iloc[:,0]
y_Categories = [item.split(' ')[0] for item in y] # pull EdemaTrue or EdemaFalse

def plotConfusion(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    tick_marks = range(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    # Add text annotations to each cell
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()

accuracies = {}
patientAccs = {}
patientNum = []
TP, TN, FP, FN = 0,0,0,0
for selectedNum in selectedNums:
    for i in range(iterations): # iterate multiple times for error bars
        patientAcc = {}
        print(f'Starting iteration: {i}')
        random_state = np.random.randint(0,4294967295) # generate random integer for random state

        # Undersample the majority class so there are equal numbers of each class
        X_edema_true = X[[item == 'EdemaTrue' for item in y_Categories]]   # Separate the data by class
        X_edema_false = X[[item == 'EdemaFalse' for item in y_Categories]]
        y_edema_true = y[[item == 'EdemaTrue' for item in y_Categories]]
        y_edema_false = y[[item == 'EdemaFalse' for item in y_Categories]]
        min_len = min(len(X_edema_true), len(X_edema_false)) # Undersample the larger class
        # print(min_len)
        # Randomly sample without replacement
        random_indices_true = np.random.choice(len(X_edema_true), min_len, replace=False)
        random_indices_false = np.random.choice(len(X_edema_false), min_len, replace=False)
        # Select the samples based on the random indices
        X_edema_true_balanced = X_edema_true[random_indices_true]
        y_edema_true_balanced = y_edema_true.iloc[random_indices_true]
        X_edema_false_balanced = X_edema_false[random_indices_false]
        y_edema_false_balanced = y_edema_false.iloc[random_indices_false]
        # Combine the balanced data
        X_balanced = np.concatenate([X_edema_true_balanced, X_edema_false_balanced], axis=0)
        y_balanced = np.concatenate([y_edema_true_balanced, y_edema_false_balanced], axis=0)
        y_Categories_balanced = [item.split(' ')[0] for item in y_balanced] # pull EdemaTrue or EdemaFalse

        # Select a subset of the original dataset
        testSize = 1 - (selectedNum / X.shape[0])
        if testSize == 0: # if we are using the entire dataset
            X1 = X_balanced
            y1 = y_balanced
        else:
            X1, _, y1, _ = train_test_split(X_balanced, y_balanced, test_size=testSize, stratify=y_Categories_balanced, random_state=random_state)

        y_Categories1 = [item.split(' ')[0] for item in y1]
        y_int = [1 if item == 'EdemaTrue' else 0 for item in y_Categories1]
        y_int = np.array(y_int)

        # find a unique set of patient IDs that are either only in test OR only in training set
        IDs = np.array([int(re.findall(r'\d+', item)[0]) for item in y1]) # Returns ID number
        uniqIDs = list(set(IDs.tolist())) # find IDs with no repeats

        j = 0
        while True: # iterate until we find a balanced dataset
            j+=1
            random_state = np.random.randint(0,4294967295) # generate random integer for random state
            uniqTrainIDs, uniqTestIDs = train_test_split(uniqIDs, test_size=0.3, random_state=random_state)

            IdTF = np.array([str(int(re.findall(r'\d+', item)[0])) +'\n ' + str('EdemaTrue' in item) for item in y1]) # returns Idnumber and if edema is present

            # use IDs to find the IDXs of each example, and then split the data by patients
            trainIDXs = []
            testIDXs = []
            for i, ID in enumerate(IDs):
                if ID in uniqTrainIDs:
                    trainIDXs.append(i)
                elif ID in uniqTestIDs:
                    testIDXs.append(i)
            X_train = X1[trainIDXs,:]
            X_test = X1[testIDXs,:]
            y_train = y_int[trainIDXs]
            y_test = y_int[testIDXs]                  
            if np.mean(y_train) > 0.47 and np.mean(y_train) < 0.53:
                if np.mean(y_test) > 0.47 and np.mean(y_test) < 0.53:
                    print(f'Balanced dataset found after {j} iterations')
                    # print(f'np.mean(y_train): {np.mean(y_train)}')
                    # print(f'np.mean(y_test): {np.mean(y_test)}')  
                    break # break if we have found a balanced dataset

        # do PCA dimensionality reduction
        pca = make_pipeline(StandardScaler(), PCA(n_components=30, random_state=random_state))
        n_neighbors = 5
        pca.fit(X_train, y_train) # fit method's model
        X_test_transformed = pca.transform(X_test)
        # # plot the PCA weights
        # plt.figure()
        # components = pca.named_steps['pca'].components_
        # wavelengths = [376.61,381.55,386.49,391.43,396.39,401.34,406.3,411.27,416.24,421.22,426.2,431.19,436.18,441.17,446.17,451.18,456.19,461.21,466.23,471.25,476.28,481.32,486.36,491.41,496.46,501.51,506.57,511.64,516.71,521.78,526.86,531.95,537.04,542.13,547.23,552.34,557.45,562.57,567.69,572.81,577.94,583.07,588.21,593.36,598.51,603.66,608.82,613.99,619.16,624.33,629.51,634.7,639.88,645.08,650.28,655.48,660.69,665.91,671.12,676.35,681.58,686.81,692.05,697.29,702.54,707.8,713.06,718.32,723.59,728.86,734.14,739.42,744.71,750.01,755.3,760.61,765.92,771.23,776.55,781.87,787.2,792.53,797.87,803.21,808.56,813.91,819.27,824.63,830,835.37,840.75,846.13,851.52,856.91,862.31,867.71,873.12,878.53,883.95,889.37,894.8,900.23,905.67,911.11,916.56,922.01,927.47,932.93,938.4,943.87,949.35,954.83,960.31,965.81,971.3,976.8,982.31,987.82,993.34,998.86,1004.39,1009.92,1015.45,1020.99,1026.54,1032.09,1037.65,1043.21]
        # for i in range(3):
        #     plt.scatter(wavelengths,components[i], label=f'Component: {i}')
        # plt.legend()
        # plt.title('PCA components')
        # plt.xlabel('Band (nm)')
        # plt.ylabel('Weight')
        # plt.show()
    

        # fit KNN on data
        # knn = KNeighborsClassifier(n_neighbors=n_neighbors) # SVM and K-means also used commonly
        # knn.fit(pca.transform(X_train), y_train) # compute the nearest neighbor classifier on the transformed training set
        # acc_knn = knn.score(pca.transform(X_test), y_test) # compute the nearest neighbor accuracy on the transformed test set
        # y_pred = knn.predict(X_test_transformed)
        # # knn.fit(X_train, y_train)
        # # acc_knn = knn.score(X_test, y_test)

        # fit SVM on data
        svc = SVC(kernel='linear')
        svc.fit(pca.transform(X_train), y_train)
        acc_svm = svc.score(X_test_transformed, y_test)  # Evaluate SVM on transformed test data
        y_pred = svc.predict(X_test_transformed)  # Pred


        # # Plot how much variance is explained by PCA
        # print(pca.named_steps['pca'].explained_variance_ratio_)
        # plt.figure(figsize=(8, 6))
        # plt.plot(range(1, len(pca.named_steps['pca'].explained_variance_ratio_) + 1), pca.named_steps['pca'].explained_variance_ratio_)
        # plt.xlabel('Number of Components')
        # plt.ylabel('Explained Variance Ratio')
        # plt.title(f'Scree Plot - Examples per iteration: {selectedNum}')
        # plt.show()

        # # Do dimnesionality reduction in the direction of categorical differences (NCA), then fit a k-nearest neighbors classifier on the transformed training set
        # nca = make_pipeline(StandardScaler(), NeighborhoodComponentsAnalysis(n_components=30, random_state=random_state))
        # n_neighbors = 3
        # knn = KNeighborsClassifier(n_neighbors=n_neighbors) # SVM and K-means also used commonly
        # nca.fit(X_train, y_train) # fit method's model
        # knn.fit(nca.transform(X_train), y_train) # compute the nearest neighbor classifier on the transformed training set
        # acc_knn = knn.score(nca.transform(X_test), y_test) # compute the nearest neighbor accuracy on the transformed test set

        # # plot the first two dimensions of the NCA
        # plt.figure()
        # X_embedded = nca.transform(X1) # transform (embed) the entire dataset 
        # scatter = plt.scatter(X_embedded[:,0],X_embedded[:,1],c=y_int,s=15,cmap="Set1", alpha=0.4, marker='.')     # (x, y, color, size, colormap) c=targets if we want a different color for each target
        # numTrain = X_train.shape[0]
        # plt.title("\n n_train={}, KNN (k={} Test accuracy = {:.2f})".format(numTrain, n_neighbors, acc_knn))
        # plt.xlabel('Component 1')
        # plt.ylabel('Compoenent 2')
        # plt.show()

        key = str(selectedNum)
        if key not in accuracies: # make a list for the overall acc
            accuracies[key] = []
        accuracies[key].append(acc_svm)

        # X_test_transformed = X_test
        isCorrect = y_pred == y_test
        # cm = plotConfusion(y_test, y_pred, ['False','True']) # plot confusion matrix
        # plt.show()

        # tally up all of the confusion matricies
        cm = confusion_matrix(y_test, y_pred)
        print(cm)  # Output: [[1 1], [1 2]]
        # Keeping a sum of all of the values. 
        TN = TN + cm[0, 0]  
        FP = FP + cm[0, 1] 
        FN = FN + cm[1, 0] 
        TP = TP + cm[1, 1] 

        isCorrect = np.where(isCorrect,1,0)
        testIDs = IdTF[testIDXs]
        for i, testID in enumerate(testIDs):
            testID = testID.item()
            if testID not in patientAcc:
                patientAcc[testID] = []
            patientAcc[testID].append(isCorrect[i].item())
        for testID in patientAcc:
            if testID not in patientAccs:
                patientAccs[testID] = []
            # patientAccs[testID].append(np.mean(np.array(patientAcc[testID])).item())
            patientAccs[testID].append(np.round(np.mean(np.array(patientAcc[testID]))).item()) # round to get the nearest threshold as the model's overall answer for for each true or false patient


        pass

        # plt.figure()
        # bars1 = plt.bar([str(key) for key in patientAcc.keys()], [patientAcc[key] for key in patientAcc], label='Accuracy')
        # plt.xlabel('Patient ID number')
        # plt.ylabel('Accuracy')
        # plt.title(f'Total Examples: {selectedNum} - Accuracy per patient')
        # plt.legend()
        # plt.show()

print(f'Overall confusion: \nTrue     false: {TN} {FP} \nLabel     true: {FN} {TP} \n     Predicted Label')
        



        


def getErrorBars(input): # do power analysis of 1D array to get 95% CI using bootstrapping
    input = np.array(input)
    avgs = np.array([])
    inputAvg = np.mean(input) # average 
    if input.size == 1:
        print('Only one input, cannot construct error bars. Setting marginOfError = 0')
        marginOfError = 0
    else:
        for _ in range(5000): # randomly choose a subset of the data with replacement and find the average 
            choices = np.random.choice(input, size=input.shape[0], replace=True)
            avg = np.mean(choices)
            avgs = np.append(avgs, avg)
        STDEV = np.std(avgs) # find the standard deviation across each of the averages
        marginOfError = (1.96*STDEV)/np.sqrt(len(input)) #z*=1.96 for a 95% confidence interval; marginOfError = ((z*)*STDEV)/sqrt(numSamples)
        CI95 = np.array([inputAvg-marginOfError, inputAvg+marginOfError]) # confidence interval = Average +- marginOfError
    return marginOfError, inputAvg

def accDict2chart(accuracies, title, xlabel): # make a bar chart with error bars from a dictionary with the accuracies for each start_ends
    marginOfErrors = []
    inputAvgs = []
    accuracies = {k: v for k, v in sorted(accuracies.items())} # sort the accuracies
    for key in accuracies: # key is the start_ends 
        epochAcc = accuracies[key] # epochAcc is a list of accuracy for each start_ends
        marginOfError, inputAvg = getErrorBars(epochAcc) # find error bars for each start_ends
        marginOfErrors.append(marginOfError)
        inputAvgs.append(inputAvg.item()) 
    keys = [str(key) for key in accuracies.keys()]
    fig, ax = plt.subplots()
    bars = ax.bar(keys, inputAvgs, yerr=marginOfErrors, capsize=5, color='skyblue', edgecolor='black') # plot bar chart with average accuracy and error bars
    ax.set_xlabel(xlabel)
    avgAcc = np.round(np.mean(np.array(inputAvgs)),2)
    ax.set_ylabel('Accuracy')
    ax.set_title(title + 'Avg Accuracy: ' + str(avgAcc))
# accDict2chart(accuracies, 'Accuracy per epoch with 95% CI', 'Number of training and testing spectra')
accDict2chart(patientAccs, f'Examples per iteration: {selectedNum} Iterations: {iterations} - Accuracy per patient 95%CI', 'Patient ID Number + Edema True/False')
plt.show()



print('All done!')