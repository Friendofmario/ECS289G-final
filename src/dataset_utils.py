# Copyright 2022 Vittorio Mazzia. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from abc import ABC, abstractmethod

import torch
import torchvision
from torchvision.datasets import MNIST, CIFAR10, EMNIST, CIFAR100
from torch.utils.data import DataLoader

import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class DatasetLoader(ABC):
    """Generic PyTorch Dataset Loader
    """

    def __init__(self):
        self.train_set = None
        self.test_set = None

        self.train_transform = None
        self.test_transform = None

        self.download_path = None

    @staticmethod
    def get_loader(dataset: torchvision.datasets, batch_size: int, shuffle: bool = True) -> DataLoader:
        """Provides a DataLoader of the given dataset

        Args:
            dataset (torchvision.datasets): input dataset
            batch_size (int): DataLoader batch size 
            shuffle (bool, optional): DataLoader shuffle option. Defaults to True.

        Returns:
            DataLoader: output DataLoader
        """
        train_loader = DataLoader(dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle)

        return train_loader

    @abstractmethod
    def download_dataset(self):
        raise NotImplementedError

    def get_train_loader(self, batch_size: int) -> DataLoader:
        return self.get_loader(self.train_set, batch_size, True)

    def get_test_loader(self, batch_size: int) -> DataLoader:
        return self.get_loader(self.test_set, batch_size, False)


class MNISTLoader(DatasetLoader):
    """MNIST PyTorch Dataset Loader
    """

    def __init__(self, train_transform: torchvision.transforms, test_transform: torchvision.transforms, download_path: str = './tmp'):
        """Initialize MNIST Dataset Loader

        Args:
            train_transform (torchvision.transforms): transformations to be applied to training set
            test_transform (torchvision.transforms): transformations to be applied to test set
            download_path (str, optional): download path. Defaults to './tmp'.
        """
        super(MNISTLoader, self).__init__()

        self.train_transform = train_transform
        self.test_transform = test_transform

        self.download_path = download_path

    def download_dataset(self) -> None:
        """Download dataset to the given path
        """
        self.train_set = MNIST(self.download_path, train=True,
                               download=True,
                               transform=self.train_transform)

        self.test_set = MNIST(self.download_path, train=False,
                              download=True,
                              transform=self.test_transform)

class CIFAR10Loader(DatasetLoader):
    """MNIST PyTorch Dataset Loader
    """

    def __init__(self, train_transform: torchvision.transforms, test_transform: torchvision.transforms, download_path: str = './tmp'):
        """Initialize MNIST Dataset Loader

        Args:
            train_transform (torchvision.transforms): transformations to be applied to training set
            test_transform (torchvision.transforms): transformations to be applied to test set
            download_path (str, optional): download path. Defaults to './tmp'.
        """
        super(CIFAR10Loader, self).__init__()

        self.train_transform = train_transform
        self.test_transform = test_transform

        self.download_path = download_path

    def download_dataset(self) -> None:
        """Download dataset to the given path
        """
        self.train_set = CIFAR10(self.download_path, train=True,
                               download=True,
                               transform=self.train_transform)

        self.test_set = CIFAR10(self.download_path, train=False,
                              download=True,
                              transform=self.test_transform)

class BreastCancerDataset(Dataset):
    def __init__(self, csv_file):
        self.data = csv_file
        # Create a StandardScaler object
        scaler = StandardScaler()
        self.features = scaler.fit_transform(self.data.drop(columns=['diagnosis', 'id', 'Unnamed: 32']).values)
        encoder = LabelEncoder()
        self.labels = encoder.fit_transform(self.data['diagnosis'].values)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        features = torch.tensor(self.features[index], dtype=torch.float32)
        label = torch.tensor(self.labels[index])
        return features, label

class BreastCancerLoader(DatasetLoader):
    def __init__(self, data_path: str = 'data/BreastCancer/breastcancer_data.csv'):
        super(BreastCancerLoader, self).__init__()
        self.download_path = data_path 

    def download_dataset(self) -> None:
        df = pd.read_csv(self.download_path)
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        self.train_set = BreastCancerDataset(train_df)
        self.test_set = BreastCancerDataset(test_df)


class WineQualityDataset(Dataset):
    def __init__(self, csv_file):
        self.data = csv_file
        # Create a StandardScaler object
        scaler = StandardScaler()
        self.features = scaler.fit_transform(self.data.values)
        encoder = LabelEncoder()
        self.labels = encoder.fit_transform(self.data['quality'].values)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        features = torch.tensor(self.features[index], dtype=torch.float32)
        label = torch.tensor(self.labels[index])
        return features, label

class WineQualityLoader(DatasetLoader):
    def __init__(self, data_path: str = 'data/WineQuality/winequality-red.csv'):
        super(WineQualityLoader, self).__init__()
        self.download_path = data_path 

    def download_dataset(self) -> None:
        df = pd.read_csv(self.download_path)
        dataset = WineQualityDataset(df)

        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        self.train_set, self.test_set = torch.utils.data.random_split(dataset, [train_size, test_size])


class EMNISTLoader(DatasetLoader):
    def __init__(self, train_transform: torchvision.transforms, test_transform: torchvision.transforms, download_path: str = './tmp'):
        """Initialize MNIST Dataset Loader

        Args:
            train_transform (torchvision.transforms): transformations to be applied to training set
            test_transform (torchvision.transforms): transformations to be applied to test set
            download_path (str, optional): download path. Defaults to './tmp'.
        """
        super(EMNISTLoader, self).__init__()

        self.train_transform = train_transform
        self.test_transform = test_transform

        self.download_path = download_path

    def download_dataset(self) -> None:
        """Download dataset to the given path
        """
        self.train_set = EMNIST(self.download_path, split="balanced", train=True,
                               download=True,
                               transform=self.train_transform)

        self.test_set = EMNIST(self.download_path, split="balanced", train=False,
                              download=True,
                              transform=self.test_transform)



class TrainingDatasetFF(torch.utils.data.Dataset):
    """Utility class to store positive and negative examples to train
       with FF algorithm.
    """

    def __init__(self, dataset_generator: DataLoader) -> None:
        """Initialize TrainingDatasetFF

        Args:
            dataset_generator (DataLoader): DataLoader to store
        """
        with torch.no_grad():
            self.dataset = [
                batch
                for X_pos, X_neg in dataset_generator
                for batch in zip(X_pos, X_neg)
            ]

    def __getitem__(self, index: int):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

class CIFAR100Loader(DatasetLoader):
    """MNIST PyTorch Dataset Loader
    """

    def __init__(self, train_transform: torchvision.transforms, test_transform: torchvision.transforms, download_path: str = './tmp'):
        """Initialize MNIST Dataset Loader

        Args:
            train_transform (torchvision.transforms): transformations to be applied to training set
            test_transform (torchvision.transforms): transformations to be applied to test set
            download_path (str, optional): download path. Defaults to './tmp'.
        """
        super(CIFAR100Loader, self).__init__()

        self.train_transform = train_transform
        self.test_transform = test_transform

        self.download_path = download_path

    def download_dataset(self) -> None:
        """Download dataset to the given path
        """
        self.train_set = CIFAR100(self.download_path, train=True,
                               download=True,
                               transform=self.train_transform)

        self.test_set = CIFAR100(self.download_path, train=False,
                              download=True,
                              transform=self.test_transform)

