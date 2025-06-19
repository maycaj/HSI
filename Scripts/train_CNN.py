import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import torchvision.models as models

class HypercubeDataset(Dataset):
    def __init__(self, file_paths, label, square_size=5):
        self.file_paths = file_paths
        self.label = label
        self.square_size = square_size

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        npy = np.load(self.file_paths[idx], allow_pickle=True).item()
        cube = npy['hyperspectral_data']
        square = cube[0:self.square_size, 0:self.square_size, :]  # Extract 5x5 square
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

    model_type = 'resnet' if use_resnet else 'cnn'
    torch.save(model.state_dict(), f'/Users/maycaj/Documents/HSI/Models/{model_type}_model.pth')
    print(f'Model training complete and saved as {model_type}_model.pth.')

if __name__ == '__main__':
    true_path = Path('/Users/maycaj/Documents/HSI/Raw Data/npyHCubesTrueCrops')
    false_path = Path('/Users/maycaj/Documents/HSI/Raw Data/npyHCubesFalseCrops')
    true_files = [file for file in true_path.iterdir() if file.is_file()]
    false_files = [file for file in false_path.iterdir() if file.is_file()]

    train_model(true_files, false_files, use_resnet=True)
