### Classify hyperspectral images using a CNN
# How to use: set the true_path and the false_path filepaths to the folders containing the numpy file of the hypercube

from pathlib import PosixPath
import numpy as np
import random
import plotly.express as px
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
from sklearn.model_selection import train_test_split  # Import train_test_split
import pandas as pd
import re

class utilities:
    @staticmethod
    def is_square_within_crop(square, npy):
        '''
        Checks to see if all of the pixels within the square region have data
        Returns (bool) whether or not the square is within the cropped region
        '''
        x, y, size = square

        # Select a square and one of the color channels
        square_pixels = npy[y:y + size, x:x + size,60]

        # Check if all of the pixels are non-zero
        inside_count = np.sum(square_pixels != 0)
        total_count = size * size
        return inside_count == total_count
    
    @staticmethod
    def get_patches(npy, frac, square_size = 5):
        '''
        Return squares from image if they are within the cropped region
        Returns:
            x0, x upper left corner
            y0: y upper left corner 
            x1: x bottom right corner
            y1: y bottom right corner 
        '''
        shape_height, shape_width = npy.shape[:2]

        # find a grid across entire image. 
        x_offset = random.randint(0, square_size - 1) #set a random offset to avoid biasing the grid
        y_offset = random.randint(0, square_size - 1)
        x_grid = np.arange(0 + x_offset, shape_width - x_offset, square_size)
        y_grid = np.arange(0 + y_offset, shape_height - y_offset, square_size)

        # find patches with data
        valid_patches = [(x0, y0, square_size) for x0 in x_grid for y0 in y_grid
                if utilities.is_square_within_crop((x0, y0, square_size), npy) == True]

        if len(valid_patches) > 10:
            valid_patches_idx = np.random.choice(len(valid_patches), int(len(valid_patches)*frac))
            valid_patches = [valid_patches[i] for i in valid_patches_idx]

        # return x,y of corners
        x0, y0, square_size = map(np.array, zip(*valid_patches))
        x1 = x0 + square_size
        y1 = y0 + square_size
        return x0, y0, x1, y1
class dataExplorer:
    @staticmethod
    def show_input(true_files, false_files, n_show=1):
        '''
        Randomly select a few images from both categories and display 
        '''
        # show the 
        npy_paths = random.sample(true_files, n_show) + random.sample(false_files, n_show)
        fig = make_subplots(rows=1, cols=len(npy_paths), subplot_titles=[f'{path.name}<br>{path.parent.name}' for path in npy_paths])
        for i, npy_path in enumerate(npy_paths):
            data = np.load(npy_path,allow_pickle=True).item()
            npy = data['hyperspectral_data']
            img = npy[:,:,[60,30,20]]
            img = (255 * (img - img.min()) / (np.ptp(img) + 1e-8)).astype(np.uint8)  # Normalize and convert to uint8
            fig.add_trace(go.Image(z=img), row=1, col=i+1,)
            x0, y0, x1, y1 = utilities.get_patches(npy, frac=0.1)

            # x0 = [x for x, _, _ in patches]
            # y0 = [y for _, y, _ in patches]
            # x1 = [x + square_size for x, _, square_size in patches]
            # y1 = [y + square_size for _, y, square_size in patches]
            fig.add_trace(
                go.Scatter(
                    x=x0,
                    y=y0,
                    mode="markers",
                    marker=dict(color="blue", size=4, opacity=1),
                    showlegend=False
                ),
                row=1, col=i+1
            )     
            fig.add_trace(
                go.Scatter(
                    x=x1,
                    y=y1,
                    mode="markers",
                    marker=dict(color="white", size=3, opacity=1),
                    showlegend=False
                ),
                row=1, col=i+1
            )             
            fig.update_layout({'title':f'{n_show*2} Sample Images where blue is the top left corner and white is the bottom left corner'})        
        fig.show()
        
class HypercubeDataset(Dataset):
    def __init__(self, file_paths, label, frac, square_size=5):
        self.file_paths = file_paths
        self.label = label
        self.square_size = square_size
        self.frac = frac
        self.samples = self._generate_samples()

    def _generate_samples(self):
        '''
        Preprocess all files to generate a list of (patch, label) samples
        '''
        samples = []
        for file_path in self.file_paths:
            npy = np.load(file_path, allow_pickle=True).item()
            cube = npy['hyperspectral_data']
            x0, y0, x1, y1 = utilities.get_patches(cube, frac=self.frac, square_size=self.square_size)
            for i in range(len(x0)):
                patch = cube[y0[i]:y1[i], x0[i]:x1[i], :]  # Extract patch
                patch = (patch - patch.min()) / (np.ptp(patch) + 1e-8)  # Normalize
                patch = np.transpose(patch, (2, 0, 1))  # Convert to (channels, height, width)
                if patch.shape != (128, self.square_size, self.square_size):
                    raise ValueError(f"Patch shape mismatch: expected (128, {self.square_size}, {self.square_size}), got {patch.shape}")
                samples.append((torch.tensor(patch, dtype=torch.float32), torch.tensor(self.label, dtype=torch.long), file_path.name, (x0[i], y0[i], x1[i], y1[i])))
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

class ConvClassify:
    def train_model(train_true, train_false, output_dir, frac, epochs=1, batch_size=16, learning_rate=0.001, use_resnet=False):
        '''
        Args:
            train_true: pathlib path names of true files
            train_false: pathlib path names of false files
            output_dir: where to output file
            epochs: number of epochs to train
            batch_size: number of training examples per step
            learning_rate: how large of a step in the direction of minimizing error
        Returns:
            model: trained model
        '''

        true_data_train = HypercubeDataset(train_true, label=1, frac=frac)
        false_data_train = HypercubeDataset(train_false, label=0, frac=frac)
        train_dataset = torch.utils.data.ConcatDataset([true_data_train, false_data_train])

        # Determine device and adjust DataLoader settings
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        pin_memory = False if device.type == "mps" else True  # Disable pin_memory for MPS

        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=4,  # Adjust based on your system
            pin_memory=pin_memory,  # Conditional pin_memory
            prefetch_factor=2,  # Prefetch batches to improve data loading speed
            drop_last=True, # Drop last batch so we don't get an error
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

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            for batch_idx, (inputs, labels, filenames, patch_loc) in enumerate(train_dataloader):
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)  # Non-blocking transfer
                optimizer.zero_grad()
                if inputs.shape[1:] != torch.Size([128, 5, 5]):
                    raise ValueError(f"Input shape mismatch: expected (batch_size, 128, 5, 5), got {inputs.shape}")
                print(inputs.shape)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            # Append epoch loss for plotting
            epoch_losses.append(epoch_loss)
            print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}')

        # Plot training loss over epochs
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, epochs + 1), epoch_losses, marker='o', label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Epochs')
        plt.legend()
        plt.grid(True)
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / "training_loss.png")  # Save the plot
        # plt.show()

        print(f'Training complete, now testing...')
        return model

    def test_model(test_true, test_false, output_dir, frac, model, batch_size=16):
        true_data_test = HypercubeDataset(test_true, label=1, frac=frac)
        false_data_test = HypercubeDataset(test_false, label=0, frac=frac)
        test_dataset = torch.utils.data.ConcatDataset([true_data_test, false_data_test])

        # Determine device and adjust DataLoader settings
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        pin_memory = False if device.type == "mps" else True  # Disable pin_memory for MPS

        test_dataloader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=4,  # Adjust based on your system
            pin_memory=pin_memory,  # Conditional pin_memory
            prefetch_factor=2  # Prefetch batches to improve data loading speed
        )

        testing_results = pd.DataFrame([])
        with torch.no_grad():
            for batch_idx, (inputs, labels, filenames, patch_loc) in enumerate(test_dataloader):
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                # You can add code here to calculate accuracy or other metrics
                testing_result = pd.DataFrame({'Batch': batch_idx+1, 
                                            'Predicted': predicted.cpu().numpy(), 
                                            'Actual': labels.cpu().numpy(),
                                            'Filenames': filenames})
                                            #    'Patch Locations': patch_loc})
                testing_results = pd.concat([testing_results, testing_result], ignore_index=True)
        testing_results['ID'] = testing_results['Filenames'].apply(lambda x: int(re.search(r'\d+',x).group()))
        testing_results['Correct'] = testing_results['Predicted'] == testing_results['Actual']
        testing_results.to_csv(output_dir / "test_outputs.csv", index=False)  # Save outputs to CSV
        print(f"Test outputs saved to {output_dir / 'test_outputs.csv'}")

        return testing_results

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

    def plot_results(testing_results):
        # Find group for processing
        group = testing_results[['Actual','ID','Correct']].groupby(['ID','Actual'])

        # Find bootstrapped values
        bootstrap_group = testing_results[['Actual','ID','Correct']].groupby(['ID','Actual'])
        bootstrap_values = bootstrap_group.apply(lambda x: pd.Series(ConvClassify.bootstrapp(x['Correct'])), 
                                                include_groups=False)
        bootstrap_values.columns = ['Lower_pct','Upper_pct']
        bootstrap_values = bootstrap_values.reset_index()

        # Find bar chart values
        chart_values = group.mean()
        chart_values = chart_values.reset_index()
        chart_values = chart_values.merge(bootstrap_values, how='left')
        chart_values['Lower_error'] = np.abs(chart_values['Lower_pct'] - chart_values['Correct'])
        chart_values['Upper_error'] = np.abs(chart_values['Upper_pct'] - chart_values['Correct'])
        chart_values['Label'] = 'ID = ' + chart_values['ID'].astype(str) + ' Category = ' + chart_values['Actual'].astype(str)

        fig = go.Figure(
            data=[
                go.Bar(
                    y=chart_values['Correct'],
                    x=chart_values['Label'],
                    error_y=dict(
                        type='data',
                        symmetric=False,
                        array = chart_values['Upper_error'],
                        arrayminus = chart_values['Lower_error'],))])
        fig.update_layout(
            title='Accuracies by ID and Category',
            xaxis_title='ID & Category (0 = EdemaFalse, 1 = EdemaTrue)',
            yaxis_title='Accuracy')
        fig.show()
        
    def ID2paths(ID_list, directories):
        '''
        Args: 
            ID_list (list of int): ID number(s) to find
            directory (list of str): filepath(s) to directory(s)
        Returns:
            file_lists: paths to every example with a matching ID. If multiple directories, will return multiple lists.
        '''
        file_lists = []
        for directory in directories:
            directory = Path(directory)
            file_list = [file for file in directory.iterdir()]

            # Select only IDs in ID_list
            file_list = [file for file in file_list if int(re.search(r'\d+',file.name).group()) in ID_list] 
            file_lists.append(file_list)
        return file_lists
    
    def run(iterations, show_preview, randomize, frac, epochs, true_path, false_path):
        # Process filenames
        pio.renderers.default = 'browser' # Make sure to set the default renderer for Plotly
        # Define paths to the true and false data directories
        true_files = [file for file in true_path.iterdir() if file.suffix == '.npy']
        false_files = [file for file in false_path.iterdir() if file.suffix == '.npy']
        true_IDs = [int(re.search(r'\d+',file.name).group()) for file in true_files]
        false_IDs = [int(re.search(r'\d+',file.name).group()) for file in false_files]
        IDs = true_IDs + false_IDs
        IDs = np.unique(IDs)

        if show_preview:
            explorer = dataExplorer()
            explorer.show_input(true_files, false_files)

        if randomize:
            random_state = np.random.randint(0, 4294967295)
        else:
            random_state = 42

        # Set random seeds for reproducibility
        torch.manual_seed(random_state)
        random.seed(random_state)
        np.random.seed(random_state)

        # If using GPU (e.g., CUDA), set the seed for CUDA as well
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_state)
            torch.cuda.manual_seed_all(random_state)
            
        # Create a timestamped folder in the Downloads directory
        timestamp = datetime.now().strftime("%Y-%m-%d %H %M")
        output_dir = Path(f'/Users/cameronmay/Downloads/CNN {timestamp}')
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save a copy of the script to the output directory
        script_path = Path(__file__)
        shutil.copy(script_path, output_dir / script_path.name)

        for i in range(iterations):
            # Select one ID for testing and use the rest for training and validation 
            ID_train_val, ID_test = train_test_split(IDs, test_size=1/len(IDs), random_state=random_state)

            # Of the training and validation set, select 20% of IDs for validation and keep the rest for training
            ID_train, ID_val = train_test_split(ID_train_val, test_size=0.20, random_state=random_state)

            # Use ID2paths() to convert from ID to filepaths 
            val_true, val_false = ConvClassify.ID2paths(ID_val, [true_path, false_path])
            train_true, train_false = ConvClassify.ID2paths(ID_train, [true_path, false_path])
            test_true, test_false = ConvClassify.ID2paths(ID_test, [true_path, false_path])
            
            # Pass filenames into train_model()
            model = ConvClassify.train_model(train_true, train_false, output_dir/str(i), frac, epochs=epochs, use_resnet=True)

            # Test model
            testing_results = ConvClassify.test_model(test_true, test_false, output_dir/str(i), frac, model)

            # Plot results of testing
            ConvClassify.plot_results(testing_results)


if __name__ == '__main__':
    # Set parameters
    parameter_list = [{
        'iterations': 2, # How many times to run the model
        'show_preview': True,  # Set to True to show the data explorer preview
        'randomize': True, # Set to True to randomize train/test split, model initialization, dataloader shuffling, and patch selection
        'frac': 0.1, # Fraction of patches to include in dataset for both training and testing. Lower percentages will run faster
        'epochs': 2, # How many epochs to run on 
        'true_path': Path('/Users/cameronmay/Documents/HSI/npy/EdemaTrue'), # Location of true numpy files
        'false_path': Path('/Users/cameronmay/Documents/HSI/npy/EdemaFalse'), # Location of false numpy files
    }]
    for parameters in parameter_list:
        ConvClassify.run(**parameters)

    print('All done')