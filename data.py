import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import copy

_DATASETS = {}

def _add_dataset(dataset_fn):
    _DATASETS[dataset_fn.__name__] = dataset_fn
    return dataset_fn


class CIFAR10WithID(datasets.CIFAR10):
    def __init__(self, *args, **kwargs):
        super(CIFAR10WithID, self).__init__(*args, **kwargs)
        # Generate a unique ID for each image based on its index
        self.ids = list(range(len(self.data)))

    def __getitem__(self, index):
        # Get the original CIFAR-10 data and label
        img, label = super(CIFAR10WithID, self).__getitem__(index)
        
        # Retrieve the unique ID for this item
        unique_id = self.ids[index]
        
        return img, label, unique_id

def _get_cifar_transforms(augment=False):
    transform_augment = transforms.Compose([
        transforms.Pad(padding=4, fill=(125,123,113)),
        transforms.RandomCrop(32, padding=0),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_train = transform_augment if augment else transform_test

    return transform_train, transform_test



@_add_dataset   
def cifar10(root, augment=False, **kwargs):
    transform_train, transform_test = _get_cifar_transforms(augment=augment)
    train_set = datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
    test_set  = datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)
    return train_set, test_set

@_add_dataset   
def cifar10withids(root, augment=False, **kwargs):
    transform_train, transform_test = _get_cifar_transforms(augment=augment)
    train_set = CIFAR10WithID(root=root, train=True, download=True, transform=transform_train)
    test_set  = CIFAR10WithID(root=root, train=False, download=True, transform=transform_test)
    return train_set, test_set

def load_data(args, **kwargs):

    train_dataset, test_dataset = _DATASETS[args.dataset](args.data_dir, args.augment, **kwargs)

    num_classes = len(set(train_dataset.targets))

    # Create DataLoader for training and testing
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    if args.val_frac == 0:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        val_loader = val_dataset = None
    else:
        train_size = int((1-args.val_frac) * len(train_dataset))
        val_size = len(train_dataset) - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    
    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset, num_classes