### Classify hyperspectral images using a CNN
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

class utilities:
    @staticmethod
    def is_square_within_crop(square, npy):
        '''
        Checks to see if all of the pixels within the square region have data
        '''
        x, y, size = square

        # Select a square and one of the color channels
        square_pixels = npy[y:y + size, x:x + size,60]

        # Check if all of the pixels are non-zero
        inside_count = np.sum(square_pixels != 0)
        total_count = size * size
        return inside_count == total_count
    
    @staticmethod
    def get_patches(npy, square_size = 5):
        '''
        Return squares from image if they are within the cropped region
        Returns:
            x0, x upper left corner
            y0: y upper left corner 
            x1: x bottom right corner
            y1: y bottom right corner 
        '''
        shape_height, shape_width = npy.shape[:2]

        # find a grid across entire image
        x_grid = np.arange(0, shape_width, square_size)
        y_grid = np.arange(0, shape_height, square_size)

        # find patches with data
        valid_patches = [(x0, y0, square_size) for x0 in x_grid for y0 in y_grid
                if utilities.is_square_within_crop((x0, y0, square_size), npy) == True]

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
            x0, y0, x1, y1 = utilities.get_patches(npy)

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
    def __init__(self, file_paths, label, square_size=5):
        self.file_paths = file_paths
        self.label = label
        self.square_size = square_size

    def __len__(self):
        return len(self.file_paths)
    
    @staticmethod
    def subdivide_npy(npy):
        '''
        divides numpy file into patches
        '''
        
        pass # ____ implement in 

    def __getitem__(self, idx):
        '''
        Loads each npy file and extracts tensors as patches
        '''
        npy = np.load(self.file_paths[idx], allow_pickle=True).item()
        cube = npy['hyperspectral_data']
        square = cube[0:self.square_size, 0:self.square_size, :]  # Extract 5x5 patch
        square = (square - square.min()) / (np.ptp(square) + 1e-8)  # Normalize
        square = np.transpose(square, (2, 0, 1))  # Convert to (channels, height, width)
        return torch.tensor(square, dtype=torch.float32), torch.tensor(self.label, dtype=torch.long)

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
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 2)  # Output layer for binary classification

    def forward(self, x):
        return self.resnet(x)

def train_model(true_files, false_files, epochs=10, batch_size=16, learning_rate=0.001, use_resnet=False):
    true_dataset = HypercubeDataset(true_files, label=1)
    false_dataset = HypercubeDataset(false_files, label=0)
    dataset = torch.utils.data.ConcatDataset([true_dataset, false_dataset])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if use_resnet:
        model = ResNet2D(input_channels=true_dataset[0][0].shape[0])
    else:
        model = CNN(input_channels=true_dataset[0][0].shape[0])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}')


if __name__ == '__main__':
    true_path = Path('/Users/maycaj/Documents/HSI/Raw Data/npyHCubesTrueCrops')
    false_path = Path('/Users/maycaj/Documents/HSI/Raw Data/npyHCubesFalseCrops')
    true_files = [file for file in true_path.iterdir() if file.is_file()]
    false_files = [file for file in false_path.iterdir() if file.is_file()]

    explorer = dataExplorer()
    explorer.show_input(true_files, false_files)

    train_model(true_files, false_files, use_resnet=True)

    print('All done')