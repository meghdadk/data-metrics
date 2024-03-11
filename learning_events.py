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
import torchvision.models as models
from torch.utils.data import Subset

import data
import models
from utils import adjust_learning_rate, manual_seed

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
    
    parser.add_argument('--num-runs', type=int, default=10, help='Number of random seeds to run')

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

def calculate_negative_entropy(probs):
    return (probs * torch.log(probs)).sum().item()

def train_model_track_event(model, train_loader, val_loader, args, device, logger=None):
    '''Deprecated!'''

    criterion = nn.CrossEntropyLoss().to(device)
    
    if args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        raise ValueError("Optimizer not supported")
    
    lr_dict = {'learning_rate': args.lr, 'lr_decay_epochs': args.lr_decay_epochs, 'lr_decay_rate': args.lr_decay_rate}

    # Stores the p_l, p_max and entropy values for each example
    softmax_confidences = defaultdict(list)
    p_max_values = defaultdict(list)
    p_e_values = defaultdict(list)
    # Stores the accuracy of the previous step for comparison
    previous_step_accuracy = {}
    # Stores the averaged P_L and the step number when a learning-event occurs
    learning_events = {}

    for epoch in range(args.epochs):
        adjust_learning_rate(epoch, lr_dict, optimizer)
        running_loss = 0
        correct_predictions = 0
        total_predictions = 0
        for i, (images, labels, unique_ids) in enumerate(train_loader):
            images, labels, unique_ids = images.to(device), labels.to(device), unique_ids.to(device)
            model.train()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            model.eval()
            # Calculate softmax probabilities and predictions
            softmax_probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            # Track accuracy and softmax confidence for each example
            for idx, unique_id in enumerate(unique_ids):
                unique_id = unique_id.item()
                if unique_id in learning_events.keys():
                    continue

                correct_class_confidence = softmax_probs[idx][labels[idx]].item()
                softmax_confidences[unique_id].append(correct_class_confidence)
                

                p_max = softmax_probs[idx].max().item()
                p_max_values[unique_id].append(p_max)
                
                p_e = calculate_negative_entropy(softmax_probs[idx])
                p_e_values[unique_id].append(p_e)

                # Initialize previous accuracy if it's the first step
                if unique_id not in previous_step_accuracy:
                    previous_step_accuracy[unique_id] = 0
                
                # Current step accuracy
                acc_i_t = 1 if predicted[idx] == labels[idx] else 0
                
                # Check for a learning-event
                if previous_step_accuracy[unique_id] == 0 and acc_i_t == 1:
                    # Compute the average softmax confidence for the correct class up until this step
                    avg_confidence = np.mean(softmax_confidences[unique_id])
                    avg_p_max = np.mean(p_max_values[unique_id])
                    avg_p_e = np.mean(p_e_values[unique_id])
                    learning_events[unique_id] = [epoch * len(train_loader) + i, avg_confidence, avg_p_max, avg_p_e]
                
                # Update previous step accuracy
                previous_step_accuracy[unique_id] = acc_i_t
    
        training_accuracy = 100 * correct_predictions / total_predictions
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {running_loss/len(train_loader):.4f}, Training Accuracy: {training_accuracy:.2f}%")
    
    return learning_events

def train_model(model, train_loader, val_loader, args, device, logger=None):

    criterion = nn.CrossEntropyLoss().to(device)
    
    if args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        raise ValueError("Optimizer not supported")
    
    lr_dict = {'learning_rate': args.lr, 'lr_decay_epochs': args.lr_decay_epochs, 'lr_decay_rate': args.lr_decay_rate}

    # Stores the p_l, p_max and entropy values for each example
    softmax_confidences = defaultdict(list)
    p_max_values = defaultdict(list)
    p_e_values = defaultdict(list)
    acc_values = defaultdict(list)
    # Stores the averaged P_L and the step number when a learning-event occurs
    learning_events = {}

    for epoch in range(args.epochs):
        adjust_learning_rate(epoch, lr_dict, optimizer)
        running_loss = 0
        correct_predictions = 0
        total_predictions = 0
        for i, (images, labels, unique_ids) in enumerate(train_loader):
            images, labels, unique_ids = images.to(device), labels.to(device), unique_ids.to(device)
            model.train()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            model.eval()
            # Calculate softmax probabilities and predictions
            softmax_probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            # Track accuracy and softmax confidence for each example
            for idx, unique_id in enumerate(unique_ids):
                unique_id = unique_id.item()

                correct_class_confidence = softmax_probs[idx][labels[idx]].item()
                softmax_confidences[unique_id].append(correct_class_confidence)
                

                p_max = softmax_probs[idx].max().item()
                p_max_values[unique_id].append(p_max)
                
                p_e = calculate_negative_entropy(softmax_probs[idx])
                p_e_values[unique_id].append(p_e)

                acc_i = 1 if predicted[idx] == labels[idx] else 0 
                acc_values[unique_id].append(acc_i)

                # Compute the average softmax confidence for the correct class up until this step
                avg_confidence = np.mean(softmax_confidences[unique_id])
                avg_p_max = np.mean(p_max_values[unique_id])
                avg_p_e = np.mean(p_e_values[unique_id])
                avg_acc = np.mean(acc_values[unique_id])
                learning_events[unique_id] = [epoch, avg_confidence, avg_p_max, avg_p_e, avg_acc]

    
        training_accuracy = 100 * correct_predictions / total_predictions
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {running_loss/len(train_loader):.4f}, Training Accuracy: {training_accuracy:.2f}%")
    
    return learning_events

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

def estimate(seed, args, logger=None):

    manual_seed(seed)

    train_loader, _, test_loader, _, _, _, num_classes = get_data(seed, args)

    model = get_model(seed, args, num_classes)
    
    results = train_model(model, train_loader, None, args, device, logger)

    # Save the model to file
    checkpoint = args.checkpoints_dir + f"event_{args.model}_{args.dataset}_seed{seed}.pt"
    torch.save(model.state_dict(), checkpoint)

    train_accuracy = evaluate_model(model, train_loader, device)
    test_accuracy = evaluate_model(model, test_loader, device)

    print(f"Train accuracy: {train_accuracy:.2f}%, Test accuracy: {test_accuracy:.2f}%")

    results = np.array([[k] + v for k, v in results.items() if len(v) > 0])
    print("Learning events shape:", results.shape)

    return dict(events=results)


if __name__ == '__main__':

    args = argparser()
    rich_print(args)
    print ('\n')

    for i in range(args.num_runs):

        npz_fn = f'results/event_results_s{i}.npz'
        if os.path.exists(npz_fn):
            estimates = np.load(npz_fn)
        else:
            metrics = estimate(seed=i, args=args, logger=None)
            np.savez(f'results/event_results_s{i}.npz', **metrics)
    