import os
import numpy as np
import yaml
import argparse
import collections
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


    # Set the h parameter for the loss curvature
    args.h = [float(i) for i in args.h.split(',')]
    if len(args.h)>args.epochs:
        raise ValueError('Length of h should be less than number of epochs')
    if len(args.h)==1:
        args.h = args.epochs * [args.h[0]]
    else:
        h_all = args.epochs * [1.0]
        h_all[:len(args.h)] = list(args.h[:])
        h_all[len(args.h):] = (args.epochs - len(args.h)) * [args.h[-1]]
        args.h = h_all

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

def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
    elif isinstance(x, collections.abc.Iterable):
        for elem in x:
            zero_gradients(elem)

def find_z(net, inputs, targets, criterion, h, device):
    ''' Adopted with modification from SOURCE: https://github.com/F-Salehi/CURE_robustness/blob/master/CURE/CURE.py
    the criterion reduction is set to non to get the loss for each sample

    Finding the direction in the regularizer
    '''
    criterion = nn.CrossEntropyLoss(reduction='none').to(device)
    inputs.requires_grad_()
    outputs = net.eval()(inputs)
    loss_z = criterion(net.eval()(inputs), targets)
             
    loss_z.backward(torch.ones(targets.size()).to(device))
    #loss_z.backward()
    grad = inputs.grad.data + 0.0
    norm_grad = grad.norm().item()
    z = torch.sign(grad).detach() + 0.
    z = 1.*(h) * (z+1e-7) / (z.reshape(z.size(0), -1).norm(dim=1)[:, None, None, None]+1e-7)  
    zero_gradients(inputs) 
    net.zero_grad()

    return z, norm_grad

def regularizer(net, inputs, targets, criterion, h, device):
    '''Adopted with modification from SOURCE: https://github.com/F-Salehi/CURE_robustness/blob/master/CURE/CURE.py
    the criterion reduction is set to non to get the loss for each sample
    also the grad_diff is not aggregated across the batch
    
    Regularizer term in CURE
    '''
    criterion = nn.CrossEntropyLoss(reduction='none').to(device)
    z, norm_grad = find_z(net, inputs, targets, criterion, h, device)
    
    inputs.requires_grad_()
    outputs_pos = net.eval()(inputs + z)
    outputs_orig = net.eval()(inputs)

    loss_pos = criterion(outputs_pos, targets)
    loss_orig = criterion(outputs_orig, targets)
    grad_diff = torch.autograd.grad((loss_pos-loss_orig), inputs, grad_outputs=torch.ones(targets.size()).to(device),
                                    create_graph=True)[0]
    

    reg = grad_diff.reshape(grad_diff.size(0), -1).norm(dim=1)
    net.zero_grad()

    #return torch.sum(reg) / float(inputs.size(0)), norm_grad
    return reg, norm_grad


def train_model(model, train_loader, val_loader, args, device, logger=None):

    criterion = nn.CrossEntropyLoss().to(device)
    
    if args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        raise ValueError("Optimizer not supported")
    
    lr_dict = {'learning_rate': args.lr, 'lr_decay_epochs': args.lr_decay_epochs, 'lr_decay_rate': args.lr_decay_rate}

    train_curvs = collections.defaultdict(list)
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

            curv, grad_norm = regularizer(model, images, labels, criterion, h=args.h[epoch], device=device)

            for _id, _c in zip(unique_ids, curv):
                train_curvs[_id.item()].append(_c.item())

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
            

        training_accuracy = 100 * correct_predictions / total_predictions
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {running_loss/len(train_loader):.4f}, Training Accuracy: {training_accuracy:.2f}%")


    # Convert train_curvs dictionary to a 2D array
    train_curvs_array = np.array([train_curvs[key] for key in sorted(train_curvs.keys())])

    # Print the shape of the array
    print("Shape of train_curvs_array:", train_curvs_array.shape)

    return np.array(train_curvs_array)   


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
    #checkpoint = args.checkpoints_dir + f"{args.model}_{args.dataset}_seed{seed}.pt"
    #torch.save(model.state_dict(), checkpoint)

    train_accuracy = evaluate_model(model, train_loader, device)
    test_accuracy = evaluate_model(model, test_loader, device)

    print(f"Train accuracy: {train_accuracy:.2f}%, Test accuracy: {test_accuracy:.2f}%")

    print (results.shape)

    return dict(curvatures=results)


if __name__ == '__main__':

    args = argparser()
    rich_print(args)
    print ('\n')


    npz_fn = f'results/loss_curvatures.npz'
    if os.path.exists(npz_fn):
        estimates = np.load(npz_fn)
    else:
        metrics = estimate(seed=args.seed, args=args, logger=None)
        np.savez(f'results/loss_curvatures.npz', **metrics)
    