### Classify hyperspectral images using a CNN
# How to use: set the true_path and the false_path filepaths to the folders containing the numpy file of the hypercube

import numpy as np
import random
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import torchvision.models as models
import plotly.io as pio
import os
from datetime import datetime
import shutil
import matplotlib.pyplot as plt  # Add matplotlib for plotting
import matplotlib
from sklearn.model_selection import train_test_split  # Import train_test_split
import pandas as pd
import re
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
import time
import json


class utilities:
    @staticmethod
    def is_square_within_crop(square, npy):
        '''
        Checks to see if all of the pixels within the square region have data
        Returns (bool) whether or not the square is within the cropped region
        '''
        x, y, size = square

        # Select a square and one of the color channels
        square_pixels = npy[y:y + size, x:x + size,30]

        # img = npy[:,:,[60,30,20]]
        # img = (255 * (img - img.min()) / (np.ptp(img) + 1e-8)).astype(np.uint8)  # Normalize and convert to uint8
        # square_pixels = img[y:y + size, x:x + size,0]

        # Check if all of the pixels are non-zero
        inside_count = np.sum(square_pixels != 0)
        total_count = size * size
        return inside_count == total_count
    
    @staticmethod
    def get_patches(npy, frac, square_size = 5, rand_offset=True):
        '''
        Return squares from image if they are within the cropped region
        Returns:
            x0, x upper left corner
            y0: y upper left corner 
            x1: x bottom right corner
            y1: y bottom right corner 
        '''
        npy_height, npy_width = npy.shape[:2]

        if rand_offset:
            #set a random offset to avoid biasing the grid
            x_offset = random.randint(0, square_size - 1) 
            y_offset = random.randint(0, square_size - 1)
        else:
            x_offset = 0
            y_offset = 0

        # find an overlapping grid across entire image. This is the basis for our searchlight
        stride = 1 # So that every location across the image can be classified
        x_grid = np.arange(0 + x_offset, npy_width - x_offset, stride)
        y_grid = np.arange(0 + y_offset, npy_height - y_offset, stride)

        # find patches with data
        valid_patches = [(x0, y0, square_size) for x0 in x_grid for y0 in y_grid
                if utilities.is_square_within_crop((x0, y0, square_size), npy) == True]

        if len(valid_patches) > 10: # Make sure there are enouch patches to select a fraction
            valid_patches_idx = np.random.choice(len(valid_patches), int(len(valid_patches)*frac), replace=False)
            valid_patches = [valid_patches[i] for i in valid_patches_idx]

        # Handle empty valid_patches case
        if not valid_patches:
            return np.array([]), np.array([]), np.array([]), np.array([])

        # return x,y of corners
        x0, y0, square_size = map(np.array, zip(*valid_patches))
        x1 = x0 + square_size
        y1 = y0 + square_size
        return x0, y0, x1, y1
    
class dataExplorer:
    @staticmethod
    def show_input(true_files, false_files, frac, n_show=1):
        '''
        Randomly select a few images from both categories and display 
        '''
        # Sample photos from the true and false categories
        npy_paths = random.sample(true_files, n_show) + random.sample(false_files, n_show)

        # Plot the randomly selected images
        fig = make_subplots(rows=1, cols=len(npy_paths), subplot_titles=[f'{path.name}<br>{path.parent.name}' for path in npy_paths])
        for i, npy_path in enumerate(npy_paths):
            data = np.load(npy_path,allow_pickle=True).item()
            npy = data['hyperspectral_data']
            img = npy[:,:,[60,30,20]]
            img = (255 * (img - img.min()) / (np.ptp(img) + 1e-8)).astype(np.uint8)  # Normalize and convert to uint8
            fig.add_trace(go.Image(z=img), row=1, col=i+1,)
            x0, y0, x1, y1 = utilities.get_patches(npy, frac=frac) #___ Change frac to the same value as what you are using for training

            # Add blue markers for top left of square
            fig.add_trace(
                go.Scatter(
                    x=x0,
                    y=y0,
                    mode="markers",
                    marker=dict(color="blue", size=4, opacity=1),
                    showlegend=True
                ),
                row=1, col=i+1
            )     
            # Add white markers for bottom right of square
            fig.add_trace(
                go.Scatter(
                    x=x1-1, # When npy is sliced, the end index iteself is not included. Thus we subtract 1 to show which patches we are including
                    y=y1-1,
                    mode="markers",
                    marker=dict(color="white", size=3, opacity=1),
                    showlegend=True
                ),
                row=1, col=i+1
            )             
            fig.update_layout({'title':f'{n_show*2} Sample Images. Squares are labeled where blue is the middle of the top left pixel and white middle of the bottom right pixel'})        
        fig.show()
    @staticmethod
    def plot_bar(test_results, output_dir, data_config):
        '''
        Args:
            test_results: dataframe with actual labels, ID numbers, correct
        Bootstrapp across iterations and make a bar chart. X: ID and foldername, Y: Accuracy with bootstrapped error bars
        '''
        # Within the same 'Actual','ID', and 'Correct' find the mean
        group = test_results[['Iteration','Actual','ID','Correct']].groupby(['ID','Actual','Iteration']).mean()
        group.reset_index(inplace=True)

        # Add a row to group which represents the entire EdemaTrue and EdemaFalse category
        categories = group.groupby('Actual').mean()
        categories.reset_index(inplace=True)
        categories['ID'] = -1 
        group = pd.concat([group,categories])

        # Add a row that represents the entire dataset. Use -1 to represent both classes
        all_IDs = pd.DataFrame({'ID': -1, 'Actual':-1, 'Iteration':-1, 'Correct': group['Correct'].mean()}, index=[0])
        group = pd.concat([group,all_IDs])

        # Bootstrap within the same 'Actual','ID', and 'Correct' which bootstraps across iterations
        bootstrap_group = group.groupby(['Actual','ID'])
        bootstrap_values = bootstrap_group.apply(lambda x: pd.Series(bootstrapp(x['Correct'])), 
                                                include_groups=False)
        bootstrap_values.columns = ['Lower_pct','Upper_pct']
        bootstrap_values.reset_index(inplace=True)

        # Find the average 'Correct' across all iterations
        chart_values = group.groupby(['Actual','ID']).mean().reset_index() # Find the mean across iterations
        chart_values = chart_values.drop('Iteration',axis=1)

        # Merge the bootstrapped values with the average 'Correct' across iterations
        chart_values = chart_values.merge(bootstrap_values, how='left')

        # Convert percentiles to the length of the error bars
        chart_values['Lower_error'] = np.abs(chart_values['Lower_pct'] - chart_values['Correct'])
        chart_values['Upper_error'] = np.abs(chart_values['Upper_pct'] - chart_values['Correct'])

        # Make labels and sort by them so that the charts are clean
        chart_values['Label'] = 'ID = ' + chart_values['ID'].astype(str) + ' Category = ' + chart_values['Actual'].astype(str)
        chart_values = chart_values.sort_values(by='Label')

        fig = go.Figure(
            data=[
                go.Bar(
                    y=chart_values['Correct'],
                    x=chart_values['Label'],
                    marker_color='grey',  # <-- Set bar color to grey
                    error_y=dict(
                        type='data',
                        symmetric=False,
                        array = chart_values['Upper_error'],
                        arrayminus = chart_values['Lower_error'],))])
        fig.update_layout(
            title=f'Accuracies by ID and Category (ID = -1 means all IDs & Category = -1 means both categories)\n{data_config}',
            xaxis_title='ID & Category (0 = EdemaFalse, 1 = EdemaTrue)',
            yaxis_title='Accuracy')
        fig.write_image(output_dir / 'bar_chart.pdf')
        fig.show()

    @staticmethod
    def plot_img_results(test_results, output_dir, save_img, square_size):
        '''
        Plots the correct (green) and incorrect (red) patches over the image
        Args: 
            test_results: Dataframe with information on x, y, and if it was correctly classified
        '''
        tested_paths = test_results['Filepath'].unique()

        if save_img:
            # If save_img, save the images overlaid with correct and incorrect predictions 
            for tested_path in tested_paths:
                data = np.load(tested_path,allow_pickle=True).item()
                npy = data['hyperspectral_data']
                img = npy[:,:,[60,30,20]]
                img = (255 * (img - img.min()) / (np.ptp(img) + 1e-8)).astype(np.uint8)  # Normalize and convert to uint8
                correct = test_results[(test_results['Filepath'] == tested_path) & (test_results['Correct'] == True)]
                incorrect = test_results[(test_results['Filepath'] == tested_path) & (test_results['Correct'] == False)]
                x0_c, y0_c, x1_c, y1_c = correct['x0'], correct['y0'], correct['x1'], correct['y1']
                x0_i, y0_i, x1_i, y1_i = incorrect['x0'], incorrect['y0'], incorrect['x1'], incorrect['y1']

                matplotlib.use('Agg') # Use non-interactive backend
                # If there are any points to plot
                if len(x0_c) != 0 or len(x0_i) != 0:
                    plt.imshow(img)

                    # If there are correct points, plot them
                    if len(x0_c) != 0:
                        for x, y in zip(x0_c, y0_c): 
                            rect = plt.Rectangle((x, y), square_size, square_size, linewidth=1,
                                                edgecolor='green', facecolor='green', alpha=0.3)
                            plt.gca().add_patch(rect)
                    # If there are incorrect points, plot them 
                    if len(x0_i) != 0:
                        for x, y in zip(x0_i, y0_i): 
                            rect = plt.Rectangle((x, y), square_size, square_size, linewidth=1,
                                                edgecolor='red', facecolor='red', alpha=0.3)
                            plt.gca().add_patch(rect)
                    plt.title(f'Img: {Path(tested_path).stem}\n Cat: {Path(tested_path).parent} \nRed means incorrect, Green means correct')
                    os.makedirs(output_dir/'maps', exist_ok=True)
                    plt.tight_layout()
                    plt.savefig(output_dir / 'maps' / Path(tested_path).stem)
                    plt.close()


        # Don't plot more than 6 images for space reasons
        tested_paths = np.random.choice(tested_paths, size=5, replace=False)

        # Find all leg accuracies
        leg_accs = test_results[['Actual','ID','Correct']].groupby(['Actual','ID']).mean()
        leg_accs.reset_index(inplace=True)

        subplot_titles = []
        for tested_path in tested_paths:
            name = Path(tested_path).name

            # Find if the example is in the edemaFalse or the edemaTrue folder
            category = Path(tested_path).parent.name
            actual = 0 if re.search('EdemaFalse', category) else 1 if re.search('EdemaTrue', category) else None
            if category == None:
                raise(ValueError('Category name is not EdemaFalse nor EdemaTrue. Should be one of those'))
            ID = int(re.search(r'\d+',name).group())

            # Find the accuracy for the current image
            leg_acc = leg_accs[(leg_accs['ID'] == ID) & (leg_accs['Actual']== actual)]
            leg_acc = f"{round(leg_acc['Correct'].values[0], 2)}"

            subplot_titles.append(f'ID: {ID} <br> Cat: {category} <br> Acc: {leg_acc}')
            pass

        fig = make_subplots(rows=1, cols=len(tested_paths), subplot_titles=subplot_titles)
        
        # Plot each of the images that were tested over and plot markers over them if they were correct
        for i, tested_path in enumerate(tested_paths):
            data = np.load(tested_path,allow_pickle=True).item()
            npy = data['hyperspectral_data']
            img = npy[:,:,[60,30,20]]
            img = (255 * (img - img.min()) / (np.ptp(img) + 1e-8)).astype(np.uint8)  # Normalize and convert to uint8
            fig.add_trace(go.Image(z=img), row=1, col=i+1,)
            correct = test_results[(test_results['Filepath'] == tested_path) & (test_results['Correct'] == True)]
            incorrect = test_results[(test_results['Filepath'] == tested_path) & (test_results['Correct'] == False)]
            x0_c, y0_c, x1_c, y1_c = correct['x0'], correct['y0'], correct['x1'], correct['y1']
            x0_i, y0_i, x1_i, y1_i = incorrect['x0'], incorrect['y0'], incorrect['x1'], incorrect['y1']

            # If correct, add green markers for top left of square
            fig.add_trace(
                go.Scatter(
                    x=x0_c,
                    y=y0_c,
                    mode="markers",
                    marker=dict(color="green", size=4, opacity=1),
                    showlegend=True
                ),
                row=1, col=i+1
            )     
            # If correct, add green markers for bottom right of square
            fig.add_trace(
                go.Scatter(
                    x=x1_c-1, # When npy is sliced, the end index iteself is not included. Thus we subtract 1 to show which patches we are including
                    y=y1_c-1,
                    mode="markers",
                    marker=dict(color="green", size=3, opacity=1),
                    showlegend=True
                ),
                row=1, col=i+1
            )     
            # If incorrect, add red markers for top left of square
            fig.add_trace(
                go.Scatter(
                    x=x0_i,
                    y=y0_i,
                    mode="markers",
                    marker=dict(color="red", size=4, opacity=1),
                    showlegend=True
                ),
                row=1, col=i+1
            )     
            # If incorrect, add red markers for bottom right of square
            fig.add_trace(
                go.Scatter(
                    x=x1_i-1, # When npy is sliced, the end index iteself is not included. Thus we subtract 1 to show which patches we are including
                    y=y1_i-1,
                    mode="markers",
                    marker=dict(color="red", size=3, opacity=1),
                    showlegend=True
                ),
                row=1, col=i+1
            )          
            fig.update_layout(
                title=f'{len(tested_paths)} Tested Images. Squares are labeled where red are the incorrect squares and green are the correct squares',
                margin=dict(t=150, b=50, l=50, r=50),  # Increase top margin for title and subplot titles
                height=600,  # Adjust overall figure height
                title_y=0.95,  # Position title higher
            )   


        fig.show()

class HypercubeDataset(Dataset):
    '''Initializes a pytorch dataset. Takes the .npy files and divides them into patches using utilities.get_patches'''
    def __init__(self, file_paths, frac, label, square_size=5):
        self.file_paths = file_paths
        self.frac = frac
        self.label = label
        self.square_size = square_size
        self.samples = self._generate_samples()

    def _generate_samples(self):
        '''
        Preprocess all files to generate a list of (patch, label) samples
        '''
        samples = []
        for file_path in self.file_paths:
            npy = np.load(file_path, allow_pickle=True).item()
            cube = npy['hyperspectral_data']
            x0, y0, x1, y1 = utilities.get_patches(cube, self.frac, square_size=self.square_size)
            for i in range(len(x0)):
                patch = cube[y0[i]:y1[i], x0[i]:x1[i], :]  # Extract patch
                patch = (patch - patch.min()) / (np.ptp(patch) + 1e-8)  # Normalize
                patch = np.transpose(patch, (2, 0, 1))  # Convert to (channels, height, width)
                if patch.shape != (128, self.square_size, self.square_size):
                    raise ValueError(f"Patch shape mismatch: expected (128, {self.square_size}, {self.square_size}), got {patch.shape}")
                samples.append((torch.tensor(patch, dtype=torch.float32), torch.tensor(self.label, dtype=torch.long), file_path.name, str(file_path), x0[i], y0[i], x1[i], y1[i]))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

class CNN(nn.Module):
    def __init__(self, input_channels):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 2 * 2, 2)  # Assuming square_size=5, output size after pooling is 2x2

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        return x

class ResNet2D(nn.Module):
    def __init__(self, input_channels):
        super(ResNet2D, self).__init__()
        self.resnet = models.resnet18()
        self.resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 2)  # Output layer for binary classification

    def forward(self, x):
        if x.shape[1:] != torch.Size([128, 5, 5]):
            raise ValueError(f"Input shape mismatch: expected (batch_size, 128, 5, 5), got {x.shape}")
        return self.resnet(x)
    
def test(test_dataloader, model, device, output_dir, fold_num, i, to_save=False):
    '''
    Test model on the testing dataset and return the result in a dataframe
    '''
    test_results = pd.DataFrame([])
    with torch.no_grad():
        for batch_idx, (inputs, labels, filenames, filepath, x0, y0, x1, y1) in enumerate(test_dataloader):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            # You can add code here to calculate accuracy or other metrics
            testing_result = pd.DataFrame({'Batch': batch_idx+1, 
                                        'Fold': fold_num,
                                        'Iteration': i,
                                        'Predicted': predicted.cpu().numpy(), 
                                        'Actual': labels.cpu().numpy(),
                                        'Filenames': filenames,
                                        'Filepath': filepath,
                                        'x0': x0,
                                        'y0': y0,
                                        'x1': x1,
                                        'y1': y1})
            test_results = pd.concat([test_results, testing_result], ignore_index=True)
    test_results['ID'] = test_results['Filenames'].apply(lambda x: int(re.search(r'\d+',x).group()))
    test_results['Correct'] = test_results['Predicted'] == test_results['Actual']
    if to_save:
        test_results.to_csv(output_dir / "test_outputs.csv", index=False)  # Save outputs to CSV
        print(f"Test outputs saved to {output_dir / 'test_outputs.csv'}")
    return test_results

def train_model(fold_num, train_IDs, test_IDs, true_files,
                                  false_files, i, frac, random_state, output_dir, square_size, show_training=False, epochs=2,
                                    batch_size=16, learning_rate=0.001, use_resnet=False):
    
    print(f'\nWorking on fold {fold_num}')
    
    # Find the files for training and testing. Return a file if its ID is in the train_IDs (or testing IDs)
    train_true = [file for file in true_files if int(re.search(r'\d+',file.name).group()) in train_IDs]
    train_false = [file for file in false_files if int(re.search(r'\d+',file.name).group()) in train_IDs]
    test_true = [file for file in true_files if int(re.search(r'\d+',file.name).group()) in test_IDs]
    test_false = [file for file in false_files if int(re.search(r'\d+',file.name).group()) in test_IDs]

    # Skip fold if either class is empty
    if len(train_true) == 0 or len(train_false) == 0:
        print(f"Skipping fold {fold_num}: Not enough samples in one of the classes. len(true)={len(train_true)} len(false)={len(train_false)}")
        return pd.DataFrame([])

    true_data_train = HypercubeDataset(train_true, frac, label=1, square_size=square_size)
    false_data_train = HypercubeDataset(train_false, frac, label=0, square_size=square_size)
    train_dataset = torch.utils.data.ConcatDataset([true_data_train, false_data_train])

    # Determine device and adjust DataLoader settings
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    pin_memory = False if device.type == "mps" else True  # Disable pin_memory for MPS

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True, # Drop last batch so we don't get an error
    )

    true_data_test = HypercubeDataset(test_true, frac=frac, label=1, square_size=square_size) # Assign 1 as the true label
    false_data_test = HypercubeDataset(test_false, frac=frac, label=0, square_size=square_size) # Assign 0 as the false label
    test_dataset = torch.utils.data.ConcatDataset([true_data_test, false_data_test])

    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
    )

    if use_resnet:
        model = ResNet2D(input_channels=true_data_train[0][0].shape[0])
        print(f'Input channels = {true_data_train[0][0].shape[0]}')
    else:
        model = CNN(input_channels=true_data_train[0][0].shape[0])
    model.to(device)  # Move model to MPS or CPU

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize lists to store epoch losses for plotting
    epoch_losses = []
    epoch_acc = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_idx, (inputs, labels, filenames, filepath, x0, y0, x1, y1) in enumerate(train_dataloader):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)  # Non-blocking transfer
            optimizer.zero_grad()
            if inputs.shape[1:] != torch.Size([128, 5, 5]):
                raise ValueError(f"Input shape mismatch: expected (batch_size, 128, 5, 5), got {inputs.shape}")
            # print(inputs.shape)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_results = test(test_dataloader, model, device, output_dir, fold_num, i, to_save=False) # outputs = model(inputs) slows down training a bit

        # Append epoch loss and accuracy for plotting
        epoch_losses.append(epoch_loss)
        epoch_acc.append(epoch_results['Correct'].mean()*100)

        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}')
        print(f'Epoch {epoch+1}/{epochs}, Test Accuracy: {epoch_results['Correct'].mean()*100}')


    # Plot training loss over epochs
    if show_training:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[i for i in range(1, epochs + 1)], y=epoch_losses, name='Epoch Loss'))
        fig.add_trace(go.Scatter(x=[i for i in range(1, epochs + 1)], y=epoch_acc, name='Epoch Accuracy (%)'))
        fig.update_layout(
            title=f'Training Loss over epochs. Fold: {fold_num}',
            xaxis_title='Epochs')
        fig.write_image(output_dir / f"training_loss_fold{fold_num}.png")
        fig.show()

    print(f'Training complete, now testing...')

    test_results = test(test_dataloader, model, device, output_dir, fold_num, i, to_save=False)
    return test_results

def bootstrapp(series, CI=95):
    series = np.array(series)
    means = []
    for i in range(1000):
        sample = random.choices(series, k=1000)
        mean = np.mean(sample)
        means.append(mean)
    means = np.array(means)
    lower_percentile = (100 - CI)/2
    upper_percentile = 100 - (100 - CI)/2
    lower_value = np.percentile(means, lower_percentile)
    upper_value = np.percentile(means, upper_percentile)
    return lower_value, upper_value

class CNN_pipeline:
    def __init__(self, true_path, false_path, show_preview, frac, random_state, n_jobs, iterations, n_splits, epochs, batch_size, save_img, square_size, show_training, data_config):
        self.true_path = true_path
        self.false_path = false_path
        self.show_preview = show_preview
        self.frac = frac
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.iterations = iterations
        self.n_splits = n_splits
        self.epochs = epochs
        self.batch_size = batch_size
        self.save_img = save_img
        self.square_size = square_size
        self.show_training = show_training
        self.data_config = data_config

    def process_data(self):
        '''
        Process files, find IDs, create output directory, set up Kfold, and show preview if requested
        Args:
            true_path: Path to the directory containing true data files
            false_path: Path to the directory containing false data files
            show_preview: Whether to show a preview of the data   
        '''

        self.start_time = time.time()    

        # Find the necissary file paths and patient IDs
        self.true_files = [file for file in self.true_path.iterdir() if file.is_file() and file.suffix == '.npy']
        self.false_files = [file for file in self.false_path.iterdir() if file.is_file() and file.suffix == '.npy']
        true_IDs = [int(re.search(r'\d+',file.name).group()) for file in self.true_files ]
        false_IDs = [int(re.search(r'\d+',file.name).group()) for file in self.false_files]

        # Filter the true_IDs using the data_config
        data_configs = {
            'Round 1: cellulitis or edemafalse': (11, 12, 15, 18, 20, 22, 23, 26, 34, 36), 
            'Round 1: peripheral or edemafalse': (1, 2, 5, 6, 7, 8, 9, 10, 13, 14, 19, 21, 24, 27, 29, 30, 31, 32, 33, 35, 37, 38, 39, 40),
            'Round 1 & 2: cellulitis or edemafalse': (11, 12, 15, 18, 20, 22, 23, 26, 34, 36, 45, 59, 61, 70),
            'Round 1 & 2: peripheral or edemafalse': (1, 2, 5, 6, 7, 8, 9, 10, 13, 14, 19, 21, 24, 27, 29, 30, 31, 32, 33, 35, 37, 38, 39, 40, 41, 42, 43, 46, 47, 48, 49, 51, 53, 54, 55, 56, 57, 58, 60, 62, 63, 64, 65, 66, 67, 68, 69, 71, 72),
            'Round 1, 2, & 3: cellulitis/edemafalse + controls': (11, 12, 15, 18, 20, 22, 23, 26, 34, 36, 45, 59, 61, 70, 73, 76, 78, 83, 84, 85, 86, 88, 89, 90, 91),
            'Round 1, 2, & 3: peripheral/edemafalse + controls': (1, 2, 5, 6, 7, 8, 9, 10, 13, 14, 19, 21, 24, 27, 29, 30, 31, 32, 33, 35, 37, 38, 39, 40, 41, 42, 43, 46, 47, 48, 49, 51, 53, 54, 55, 56, 57, 58, 60, 62, 63, 64, 65, 66, 67, 68, 69, 71, 72, 74, 75, 77, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90, 91), 
        }
        true_IDs = [ID for ID in true_IDs if ID in data_configs[self.data_config]]
        IDs = true_IDs + false_IDs
        self.IDs = np.unique(IDs)

        # Create a timestamped folder in the Downloads directory
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H %M")
        self.output_dir = Path(f'/Users/cameronmay/Downloads/CNN {self.timestamp}')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save parameters in .txt file
        parameters_file = self.output_dir / 'parameters.txt'
        serializable_dict = {k: str(v) if isinstance(v, Path) else v for k, v in parameter_dict.items()} # Convert posixpaths to str so we can save
        with open(parameters_file, 'w') as f:
            json.dump(serializable_dict, f, indent=4)

        # Save a copy of the script to the output directory
        script_path = Path(__file__)
        shutil.copy(script_path, self.output_dir / script_path.name)

        if self.n_splits == 'ID':
            self.n_splits = len(self.IDs)
        self.kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        # Make sure to set the default renderer for Plotly. Show a preview of the patches that will be used in the analysis
        pio.renderers.default = 'browser'
        if self.show_preview:
            print('Showing preview...')
            dataExplorer.show_input(self.true_files, self.false_files, self.frac)

    def train(self):
        '''
        Train the CNN and output test_results (Location, ID, Actual, Predicted, Correct, x0, y0, x1, y1)
        '''
        self.test_results = pd.DataFrame([])
        for i in range (self.iterations):
            print(f'\n\n---------Working on iteration {i}---------')
            # process each fold in parallel
            test_results_fold = Parallel(n_jobs=self.n_jobs, backend='loky')( 
                delayed(train_model)(fold_num, self.IDs[train_index], self.IDs[test_index], self.true_files,
                                    self.false_files, i, self.frac, self.random_state, self.output_dir, square_size=self.square_size, show_training=self.show_training, epochs=self.epochs,
                                        batch_size=self.batch_size, learning_rate=0.001, use_resnet=False)
                for fold_num, (train_index, test_index) in enumerate(self.kf.split(self.IDs))
            )
            test_results_iteration = pd.concat(test_results_fold, ignore_index=True)

            # Add all of testing results one dataframe
            self.test_results = pd.concat([self.test_results, test_results_iteration])        # test_results = train_model(train_true, train_false, test_true, test_false, frac, use_resnet=True)

    def evaluate(self):
        '''Evaluate the model and save results'''

        print('Saving and plotting results...')
        self.test_results.to_csv(self.output_dir / 'test_outputs.csv')
        print(f'Test results saved to {self.output_dir / 'test_outputs.csv'}')
        dataExplorer.plot_bar(self.test_results, self.output_dir, self.data_config)
        dataExplorer.plot_img_results(self.test_results, self.output_dir, self.save_img, self.square_size)
        end_time = time.time()
        print(f'Done with Training \n Total time: {end_time - self.start_time}')

    def run(self):
        ''' Main method to execute the entire pipeline'''
        self.process_data()
        self.train()
        self.evaluate()


if __name__ == '__main__':
    parameter_dicts = [{'true_path': Path('/Users/cameronmay/Documents/HSI/npy/Round 3 all/EdemaTrueCrops RGB 7-14-2025Npy'), 
                  'false_path': Path('/Users/cameronmay/Documents/HSI/npy/Round 3 all/EdemaFalseCrops RGB 7-14-2025Npy'), 
                  'show_preview': True, # Set to True to show the data explorer preview
                  'frac': 0.01, # Fraction of squares to include in the analysis
                  'random_state': np.random.randint(0, 4294967295), # Seed to randomize the input. For reproducability use a number like 42, otherwise use np.random.randint(0, 4294967295)
                  'n_jobs': -1, # -1 uses as many CPUs as there are available. Positive numbers reflect the number of CPUs used
                  'iterations': 3, # Number of iterations to run the entire model 
                  'n_splits': 'ID', # 'ID' if the number of splits are the same as the number of IDs. Otherwise provide a number greater than 2
                  'epochs': 10, # Number of epochs to train the model
                  'batch_size': 16, # Batch size for training
                  'save_img': True, # Whether or not to save the images with correct and incorrect predictions
                  'square_size': 5, # Size of the square patches to extract from the hyperspectral images
                  'show_training': True, # Whether or not to plot accuracy over epochs
                  'data_config': 'Round 1, 2, & 3: peripheral/edemafalse + controls', # What data to include. The keys of data_configs are the options for what can be put here.
                  }] 

    for parameter_dict in parameter_dicts:
        instance = CNN_pipeline(**parameter_dict)
        instance.run()  # Run the CNN pipeline

    print('All done')