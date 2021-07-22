'''
TODO:

1. Import necassary libraries
2. Define all paths
3. Define DataLoader 

'''

# required imports: torch, torchvision, transforms, dataloader, datasets

import torch
from torch.utils.data import DataLoader


import torchvision
from torchvision.datasets.folder import ImageFolder
import torchvision.transforms as transforms
import torchvision.datasets as datasets

class PreProcessor():

    def __init__(self, size, mean, std, BATCH, shuffle=True, path=False):
        self.size = size
        self.mean = mean
        self.std = std
        self.path = path
        self.BATCH = BATCH
        self.shuffle = shuffle

    def train_transformer(self):
        return transforms.Compose([
            transforms.Resize([self.size,self.size]),
            transforms.ToTensor(),
            transforms.Normalize(
                mean = (self.mean, self.mean, self.mean), 
                std = (self.std, self.std, self.std)
                )
        ])

    def test_transformer(self):
        return transforms.Compose([
            transforms.Resize([self.n, self.n]),
            transforms.ToTensor(),
            transforms.Normalize(
                mean = (self.mean, self.mean, self.mean),
                std = (self.std, self.std, self.std)
            )
        ])

    def train_loader(self, train_root):

        if self.path:
            
            train_dataset = ImageFolderWithPaths(
                root = train_root,
                transform = self.train_transformer()
            )
            
            return DataLoader(
                dataset = train_dataset,
                batch_size = self.BATCH,
                shuffle = self.shuffle
            )
        
        else:
            
            train_dataset = datasets.ImageFolder(
                root = train_root,
                transform = self.train_transformer()
            )

            return DataLoader(
                dataset = train_dataset,
                batch_size = self.BATCH,
                shuffle = self.shuffle
            )
    
    def test_loader(self, test_root):

        test_dataset = datasets.ImageFolder(
            root = test_root,
            transform = self.test_transformer()
        )

        return DataLoader(
            dataset = test_dataset,
            batch_size = self.BATCH,
            shuffle = self.shuffle
        )

class ImageFolderWithPaths(torchvision.datasets.ImageFolder):

    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path