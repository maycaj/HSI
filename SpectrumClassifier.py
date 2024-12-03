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
filepath = '/Users/maycaj/Documents/HSI_III/spectrums-12-1-24_EdemaTF_imgNum.csv'
selectedNums = [3000] # max is 49652
iterations = 100

X = np.genfromtxt(filepath, delimiter=',') #read in the .csv as a npy variable
X = X[1:,1:] # get rid of y column and column labels
y = pd.read_csv(filepath) # pull y column
y = y.iloc[:,0]
y_Categories = [item.split(' ')[0] for item in y] # pull EdemaTrue or EdemaFalse

def makeConfusion(y_true, y_pred, class_names):
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
        knn = KNeighborsClassifier(n_neighbors=n_neighbors) # SVM and K-means also used commonly
        pca.fit(X_train, y_train) # fit method's model
        knn.fit(pca.transform(X_train), y_train) # compute the nearest neighbor classifier on the transformed training set
        acc_knn = knn.score(pca.transform(X_test), y_test) # compute the nearest neighbor accuracy on the transformed test set
        # knn.fit(X_train, y_train)
        # acc_knn = knn.score(X_test, y_test)


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
        accuracies[key].append(acc_knn)

        X_test_transformed = pca.transform(X_test)
        # X_test_transformed = X_test
        y_pred = knn.predict(X_test_transformed)
        isCorrect = y_pred == y_test
        # cm = makeConfusion(y_test, y_pred, ['False','True']) # plot confusion matrix
        # plt.show()

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

        



        


def getErrorBars(input): # do power analysis of 1D array to get 95% CI using bootstrapping
    input = np.array(input)
    avgs = np.array([])
    inputAvg = np.mean(input) # average 
    if input.size == 1:
        print('Only one input, cannot construct error bars. Setting marginOfError = 0')
        marginOfError = 0
    else:
        for _ in range(1000): # randomly choose a subset of the data with replacement and find the average 
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