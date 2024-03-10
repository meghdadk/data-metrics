import os
import numpy as np
import yaml
from tqdm import tqdm
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
from rich import print as rich_print

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler as SubsetSampler
import torchvision.models
from torch.utils.data import Subset

import data
import models
from utils import adjust_learning_rate, manual_seed
from scipy.stats import spearmanr


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_config(file_path, exp_key):
    with open(file_path, "r") as file:
        all_configs = yaml.safe_load(file)
        configs = all_configs.get(exp_key)
        return argparse.Namespace(**configs)
         

def argparser():
    # Argument parsing
    parser = argparse.ArgumentParser(description='Train a model on a specified dataset with various hyperparameters.')
    parser.add_argument('--config', type=str, default=None, help='Path to the configuration YAML file')
    parser.add_argument('--exp-key', type=str, default=None, help='Key for the experiment configuration')
    
    parser.add_argument('--train', action='store_true', help='Whether to train the model or load checkpoints')
    parser.add_argument('--num-runs', type=int, default=10, help='Number of shadow models')
    parser.add_argument('--subset-ratio', type=float, default=0.7, help='Fraction of seen vs unseen data to use for training')

    parser.add_argument('--dataset', type=str, default='cifar10', help='Name of the dataset to use')
    parser.add_argument('--augment', action='store_true', help='Whether to use data augmentation')
    parser.add_argument('--val-frac', type=float, default=0.2, help='Fraction of training data to use for validation')
    parser.add_argument('--data-dir', type=str, default='/data/image_data/cifar10/', help='Folder containing the dataset files')
    parser.add_argument('--model', type=str, default='resnet101', help='Name of the model to use (resnet18, resnet50, resnet101)')
    parser.add_argument('--weights-init', type=str, default='random', help='Pretrained weights to use (imagenet, random)')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train for')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--optimizer', type=str, default='sgd', help='Optimizer to use (Adam, SGD)')
    parser.add_argument('--criterion', type=str, default='ce', help='criteria to use (ce, mse)')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay (L2 penalty)')
    parser.add_argument('--lr-decay-rate', type=float, default=0.1, help='Learning rate decay rate (gamma)')
    parser.add_argument('--lr-decay-epochs', type=str, default='15,20,25', help='Comma-separated string of epochs after which to decay the learning rate')
    parser.add_argument('--checkpoints-dir', type=str, default='checkpoints/', help='Relative directory to save model/data checkpoints')


    #Wandb arguments
    parser.add_argument('--wandb-mode', type=str, default='disabled', choices=['online', 'offline', 'disabled'], 
                        help='wandb running mode')
    parser.add_argument('--wandb-project', type=str, default=None,
                        help='the project on wandb to add the runs')
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help='your wandb user name')
    parser.add_argument('--wandb-run-id', type=str, default=None,
                        help='To resume a previous run with an id')
    parser.add_argument('--wandb-group-name', type=str, default=None,
                        help='Given name to group runs together')


    args = parser.parse_args()
    if args.config is not None:
        assert args.exp_key is not None, "Please provide the experiment name"

        config = load_config(args.config, args.exp_key)
        args = config
    
    # Convert lr_decay_epochs from string to list of integers
    args.lr_decay_epochs = [int(epoch) for epoch in args.lr_decay_epochs.split(',')]

    return args


def get_data(seed, args):
    manual_seed(seed) # Reset seed for consistency in training
    
    # Load data
    (train_loader, val_loader, test_loader, 
        train_dataset, val_dataset, test_dataset, num_classes) = data.load_data(args=args)
    

    print(f"Loaded {args.dataset} dataset with {num_classes} classes")
    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset, num_classes


def get_model(seed, args, num_classes):
    manual_seed(seed)  # Reset seed for consistency in training

    # Load model
    model = models.load_model(args.model, seed, num_classes, weights=args.weights_init).to(device)

    return model


def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target, _ in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = 100 * correct / total
    return accuracy

def load_memorization(path):
    loaded_results = np.load(path)
    loaded_memorization = loaded_results['memorization']
    
    return loaded_memorization

def get_flow(model, train_loader):
    activation_indices = []  # Global list to store activation indices for the current data point
    per_layer_num_activations = []

    def hook_fn(module, input, output):
        # Use torch.argmax to find the indices of the maximum values across the channel dimension
        # This will result in a tensor of shape [batch_size, h, w]
        per_layer_num_activations.append(output.shape[1])
        #max_indices = torch.argmax(output, dim=1)
        max_indices = torch.topk(output, k=int(output.shape[1]/2), dim=1).indices
        # Store the max_indices tensor directly; no need to convert to list here
        activation_indices.append(max_indices.cpu().numpy())


    layers = [model.relu, model.layer1[0].relu, model.layer1[1].relu, model.layer2[0].relu, model.layer2[1].relu, 
            model.layer3[0].relu, model.layer3[1].relu, model.layer4[0].relu, model.layer4[1].relu]


    for layer in layers:
        layer.register_forward_hook(hook_fn)

    information_flow_dict = {}  # Dictionary to store unique_ids and their corresponding activation indices

    for images, labels, unique_ids in tqdm(train_loader):
        images, labels, unique_ids = images.to(device), labels.to(device), unique_ids.to(device)
        activation_indices.clear()  # Clear the list for the current batch
        per_layer_num_activations.clear()
        output = model(images)  # Forward pass for the whole batch


        # due to the residula connections, relu is called twice for each layer (except for the first relu). we only need the last one
        activation_indices = activation_indices[::2]
        per_layer_num_activations = per_layer_num_activations[::2]


        # Assuming unique_ids is a tensor, convert it to a list for indexing
        unique_ids_list = unique_ids.tolist()
        
        # Now, activation_indices will have a shape [batch_size, num_layers]
        # where each row corresponds to a data point and each column to a layer
        for i, indices_per_layer in enumerate(zip(*activation_indices)):
            # For each data point in the batch, store its activation indices
            information_flow_dict[unique_ids_list[i]] = list(indices_per_layer)


    # Flatten and concatenate the arrays for each key in the activation_indices dictionary
    per_layer_num_activations = np.cumsum([0] + per_layer_num_activations)
    for i, key in enumerate(information_flow_dict):
        information_flow_dict[key] = np.concatenate([arr.flatten()+per_layer_num_activations[j] for j, arr in enumerate(information_flow_dict[key])])

    sorted_dict = sorted(information_flow_dict.items(), key=lambda x: x[0])
    information_flow_array = np.array([value for _, value in sorted_dict])


    return information_flow_array, per_layer_num_activations

def estimate(seed, args, logger=None):
    manual_seed(seed)

    train_loader, _, test_loader, _, _, _, num_classes = get_data(seed, args)
    train_loader = DataLoader(train_loader.dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = get_model(seed, args, num_classes)

    checkpoint = args.checkpoints_dir + f"{args.model}_{args.dataset}_seed{seed}.pt"
    
    if os.path.exists(checkpoint):
        model.load_state_dict(torch.load(checkpoint))
        print("Checkpoint loaded successfully")
    else:
        raise ValueError("Checkpoint not found")

    model.eval()

    info_flow, per_layer_num_activations = get_flow(model, train_loader)
    memorization = load_memorization('checkpoints/estimates_results.npz')

    # Sort memorizations in ascending order
    class_indices = np.where(np.array(train_loader.dataset.targets) == 0)
    class_memorization = memorization[class_indices]
    class_info_flow = info_flow[class_indices]
    sorted_indices = np.argsort(class_memorization)
    sorted_memorization = memorization[sorted_indices]

    # Select the two largest indices
    largest_indices = sorted_indices[-10:]

    # Find the corresponding rows in info_flow
    rows = class_info_flow[largest_indices]
    row1 = rows[0]
    row2 = rows[9]
    

    buckets = per_layer_num_activations
    overlaps = []
    for i in range(len(buckets) - 1):
        start = buckets[i]
        end = buckets[i+1]
        bucket1 = row1[start:end]
        bucket2 = row2[start:end]
        overlap = len(set(bucket1) & set(bucket2)) / len(set(bucket1) | set(bucket2))
        overlaps.append(overlap)
    
    print (overlaps)
    print (class_memorization[largest_indices])
    average_overlap = sum(overlaps) / (len(buckets) - 1)
    print(f"Average overlap: {average_overlap}")
    exit()

    pairwise_comparisons = {}
    for cls in set(list(train_loader.dataset.targets)):

        class_indices = np.where(np.array(train_loader.dataset.targets) == cls)

        class_memorization = memorization[class_indices]
        class_info_flow = info_flow[class_indices]

        num_zeros = []
        for i in range(len(class_info_flow)):
            zeros = []
            for j in range(len(class_info_flow)):

                diff = np.abs(class_info_flow[i] - class_info_flow[j])
                zeros.append(np.count_nonzero(diff == 0))
            num_zeros.append(zeros)
        num_zeros = np.array(num_zeros)
        num_zeros_avg = np.max(num_zeros, axis=1)
        pairwise_comparisons[str(cls)] = num_zeros
        #sorted_indices = np.argsort(class_memorization)
        #num_zeros_sorted = num_zeros_avg[sorted_indices]

        print (f"Class {cls}")

    np.savez('checkpoints/pairwise_comparisons.npz', **pairwise_comparisons)


if __name__ == '__main__':

    args = argparser()
    rich_print(args)
    args.batch_size = 1024
    print ('\n')

    estimate(seed=42, args=args)
    

    