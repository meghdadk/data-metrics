import os
import numpy as np
import yaml
from tqdm import tqdm
import argparse
import copy

from rich import print as rich_print

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

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
    
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
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
    parser.add_argument('--checkpoint-file', type=str, default=None, help='Relative directory to save model/data checkpoints')
    


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

def calculate_kl(p, q):
    kl_divergence = (p * (torch.log(p) - torch.log(q))).sum(dim=1) + (q * (torch.log(q) - torch.log(p))).sum(dim=1)
    kl_divergence = kl_divergence / 2
    return kl_divergence

def finetune(model, train_loader, args, device, logger=None):

    criterion = nn.CrossEntropyLoss().to(device)
    
    if args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        raise ValueError("Optimizer not supported")
    
    #lr_dict = {'learning_rate': args.lr, 'lr_decay_epochs': args.lr_decay_epochs, 'lr_decay_rate': args.lr_decay_rate}

    best_loss = float('inf')
    epochs_without_improvement = 0
    for epoch in range(args.epochs):
        #adjust_learning_rate(epoch, lr_dict, optimizer)
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


        # Check if the running_loss has improved
        if running_loss < best_loss:
            best_loss = running_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Check if training should stop
        if epochs_without_improvement >= 4:
            print(f"No improvement in loss for {epochs_without_improvement} epochs. Stopping training.")
            break

        training_accuracy = 100 * correct_predictions / total_predictions
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {running_loss/len(train_loader):.4f}, Training Accuracy: {training_accuracy:.2f}%")
    
    return 0


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

def retrain_divergences(original_model, retrain_model, train_loader, args, device):

    train_loader = torch.utils.data.DataLoader(train_loader.dataset, batch_size=args.batch_size, shuffle=False)

    original_model.eval()
    retrain_model.eval()
    kl_divergences = []
    with torch.no_grad():
        for images, labels, _ in train_loader:
            images = images.to(device)
            orig_probs = torch.softmax(original_model(images), dim=1)
            retrain_probs = torch.softmax(retrain_model(images), dim=1)
            kl_divergence = calculate_kl(orig_probs, retrain_probs)
            kl_divergences.extend(kl_divergence.cpu().numpy().flatten())
    
    kl_divergences = np.array(kl_divergences)
   
    return kl_divergences

def estimate(seed, args, logger=None):

    manual_seed(seed)

    train_loader, _, test_loader, _, _, _, num_classes = get_data(seed, args)

    model = get_model(seed, args, num_classes)

    checkpoint = args.checkpoint_file
    
    if os.path.exists(checkpoint):
        model.load_state_dict(torch.load(checkpoint))
        print("Checkpoint loaded successfully")
    else:
        raise ValueError("Checkpoint not found")


    retrain = copy.deepcopy(model)
    finetune(retrain, test_loader, args, device, logger)

    orig_train_accuracy = evaluate_model(model, train_loader, device)
    orig_test_accuracy = evaluate_model(model, test_loader, device)

    ret_train_accuracy = evaluate_model(retrain, train_loader, device)
    ret_test_accuracy = evaluate_model(retrain, test_loader, device)

    print(f"Train accuracy: {orig_train_accuracy:.2f}%, Test accuracy: {orig_test_accuracy:.2f}%")
    print(f"Retrain Train accuracy: {ret_train_accuracy:.2f}%, Retrain Test accuracy: {ret_test_accuracy:.2f}%")   



    kl_divergences = retrain_divergences(model, retrain, train_loader, args, device)

    print("KL Divergences shape: ", kl_divergences.shape)

    return dict(kl_divergences=kl_divergences)



if __name__ == '__main__':

    args = argparser()
    rich_print(args)
    print ('\n')


    npz_fn = f'results/heldout_retrain.npz'
    if os.path.exists(npz_fn):
        estimates = np.load(npz_fn)
    else:
        metrics = estimate(seed=args.seed, args=args, logger=None)
        np.savez(f'results/heldout_retrain.npz', **metrics)

    