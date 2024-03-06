import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from itertools import cycle

def manual_seed(seed):
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def adjust_learning_rate(epoch, lr_dict, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(lr_dict['lr_decay_epochs']))
    new_lr = lr_dict['learning_rate']
    if steps > 0:
        new_lr = lr_dict['learning_rate'] * (lr_dict['lr_decay_rate'] ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
    return new_lr

def batch_correctness(model, dataset, device, batch_size=1024):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    correctness_list = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correctness_list.append(predicted == targets)

    return torch.cat(correctness_list).detach().cpu().numpy()



def train_model(model, train_loader, val_loader, args, device, logger=None):
    if args.criterion.lower() == 'ce':
        criterion = nn.CrossEntropyLoss().to(device)
    elif args.criterion.lower() == 'mse':
        criterion = nn.MSELoss().to(device)
    elif args.criterion.lower() == 'cosine':
        criterion = nn.CosineEmbeddingLoss().to(device)
    else:
        raise ValueError("Criterion not supported")
    
    if args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        raise ValueError("Optimizer not supported")
    
    lr_dict = {'learning_rate': args.lr, 'lr_decay_epochs': args.lr_decay_epochs, 'lr_decay_rate': args.lr_decay_rate}
    for epoch in range(args.epochs):
        model.train()
        adjust_learning_rate(epoch, lr_dict, optimizer)
        running_loss = 0
        correct_predictions = 0
        total_predictions = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if args.criterion.lower() == 'ce':
                _, predicted = torch.max(outputs.data, 1)
                total_predictions += targets.size(0)
                correct_predictions += (predicted == targets).sum().item()
            elif args.criterion.lower() == 'mse':
                total_predictions += targets.numel()
                correct_predictions += ((outputs - targets) ** 2).sum().item()

        training_accuracy = 100 * correct_predictions / total_predictions

        log_dict = {"loss": running_loss/len(train_loader), "learning_rate": optimizer.param_groups[0]['lr'], "training_accuracy": training_accuracy}
        if val_loader is not None:
            model.eval()
            val_accuracy = evaluate_model(model, val_loader, device)
            log_dict["val_accuracy"] = val_accuracy
            print(f"Epoch {epoch+1}/{args.epochs}, Loss: {running_loss/len(train_loader):.4f}, Training Accuracy: {training_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%")
        else:
            print(f"Epoch {epoch+1}/{args.epochs}, Loss: {running_loss/len(train_loader):.4f}, Training Accuracy: {training_accuracy:.2f}%")

        if logger is not None:
            logger.log(log_dict)
            
        

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = 100 * correct / total
    return accuracy