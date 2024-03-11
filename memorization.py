import os
import numpy as np
import yaml
from tqdm import tqdm
import argparse
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
from utils import train_model, adjust_learning_rate, batch_correctness, manual_seed

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

def split_data(seed, train_loader, subset_ratio, batch_size):

    manual_seed(seed)

    num_train_total = len(train_loader.dataset)
    num_train = int(num_train_total * subset_ratio)
    num_batches = int(np.ceil(num_train / batch_size))

    subset_sampler = SubsetSampler(np.random.choice(num_train_total, size=num_train, replace=False))
    sub_train_loader = DataLoader(train_loader.dataset, batch_size=batch_size, sampler=subset_sampler)
    
    return num_train_total, subset_sampler, sub_train_loader

def get_model(seed, args, num_classes):
    manual_seed(seed)  # Reset seed for consistency in training

    # Load model
    model = models.load_model(args.model, seed, num_classes, weights=args.weights_init).to(device)

    return model


def save_seed_indices(seed, args):

    manual_seed(seed)

    train_loader, _, _, _, _, _, _ = get_data(seed, args)
    
    num_train_total, subset_sampler, sub_train_loader = split_data(seed, train_loader, args.subset_ratio, args.batch_size)

    trainset_mask = np.zeros(num_train_total, dtype=np.bool_)
    trainset_mask[subset_sampler.indices] = True

    if not os.path.exists(args.checkpoints_dir):
        os.makedirs(args.checkpoints_dir)
    
    np.save(os.path.join(args.checkpoints_dir, 'resnet18_cifar10_indices{}.npy'.format(seed)), trainset_mask)
    
def subset_train(seed, args, logger=None):

    manual_seed(seed)

    train_loader, _, test_loader, _, _, _, num_classes = get_data(seed, args)

    model = get_model(seed, args, num_classes)

    num_train_total, subset_sampler, sub_train_loader = split_data(seed, train_loader, args.subset_ratio, args.batch_size)

    if not args.train:
        chkpt_path = os.path.join(args.checkpoints_dir, 'resnet18_cifar10_model{}.pt'.format(seed))
        assert os.path.exists(chkpt_path), f"Checkpoint file {chkpt_path} does not exist"
        model.load_state_dict(torch.load(chkpt_path))
    else:
        train_model(model, sub_train_loader, None, args, device, logger)


    model.eval()


    trainset_correctness = batch_correctness(model, train_loader.dataset, device)
    testset_correctness = batch_correctness(model, test_loader.dataset, device)

    # Create a subset of the dataset for the selected indices
    subset_dataset = Subset(train_loader.dataset, subset_sampler.indices)
    sub_train_loader = DataLoader(subset_dataset, batch_size=args.batch_size)

    # Create a DataLoader for the left-out subsample
    left_out_indices = np.setdiff1d(np.arange(num_train_total), subset_sampler.indices)
    left_out_dataset = Subset(train_loader.dataset, left_out_indices)
    left_out_loader = DataLoader(left_out_dataset, batch_size=args.batch_size)


    trainset_mask = np.zeros(num_train_total, dtype=np.bool_)
    trainset_mask[subset_sampler.indices] = True

    # Compute accuracy on the subsamples
    selected_subset_correctness = batch_correctness(model, sub_train_loader.dataset, device)
    left_out_subset_correctness = batch_correctness(model, left_out_loader.dataset, device)

    print ("sub_train_loader.dataset: ", len(sub_train_loader.dataset), "left_out_loader.dataset: ", len(left_out_loader.dataset))

    # Print accuracies
    print(f"Selected Subset Train Accuracy: {np.mean(selected_subset_correctness):.4f}")
    print(f"Left-Out Subset Train Accuracy: {np.mean(left_out_subset_correctness):.4f}")
    print(f"Test Accuracy: {np.mean(testset_correctness):.4f}")

    if args.train:
        if not os.path.exists(args.checkpoints_dir):
            os.makedirs(args.checkpoints_dir)

        chkpt_path = os.path.join(args.checkpoints_dir, 'resnet18_cifar10_model{}.pt'.format(seed))
        torch.save(model.state_dict(), chkpt_path)


    return trainset_mask, trainset_correctness, testset_correctness


def estimate(args, logger=None):
    results = []

    for i_run in range(0, args.num_runs,1):
        save_seed_indices(i_run, args)
        results.append(subset_train(i_run, args, logger))

    trainset_mask = np.vstack([ret[0] for ret in results])
    inv_mask = np.logical_not(trainset_mask)
    trainset_correctness = np.vstack([ret[1] for ret in results])
    testset_correctness = np.vstack([ret[2] for ret in results])


    def _masked_avg(x, mask, axis=0, esp=1e-10):
        return (np.sum(x * mask, axis=axis) / np.maximum(np.sum(mask, axis=axis), esp)).astype(np.float32)

    def _masked_dot(x, mask, esp=1e-10):
        x = x.T.astype(np.float32)
        return (np.matmul(x, mask) / np.maximum(np.sum(mask, axis=0, keepdims=True), esp)).astype(np.float32)

    mem_est = _masked_avg(trainset_correctness, trainset_mask) - _masked_avg(trainset_correctness, inv_mask)
    c_scores = _masked_avg(trainset_correctness, inv_mask)
    infl_est = _masked_dot(testset_correctness, trainset_mask) - _masked_dot(testset_correctness, inv_mask)

    print(f'Avg test acc = {np.mean(testset_correctness):.4f} ± {np.std(testset_correctness):.4f}')
    print("memory array shape: ", mem_est.shape, "influence shape: ", infl_est.shape)

    return dict(memorization=mem_est, influence=infl_est, c_scores=c_scores)


def show_examples(estimates, args, n_show=10):
    cifar10_train_loader, _, cifar10_test_loader, _, _, _, num_classes = get_data(seed=0, args=args)
    def show_image(ax, image, title=None):
        ax.axis('off')
        ax.imshow(image)
        if title is not None:
            ax.set_title(title, fontsize='x-small')

    n_context1 = 4
    n_context2 = 5

    fig, axs = plt.subplots(nrows=n_show, ncols=n_context1 + n_context2 + 1,
                            figsize=(n_context1 + n_context2 + 1, n_show))
    idx_sorted = np.argsort(np.max(estimates['influence'], axis=1))[::-1]
    for i in range(n_show):
        # show test example
        idx_tt = idx_sorted[i]
        label_tt = cifar10_test_loader.dataset.targets[idx_tt]
        show_image(axs[i, 0], cifar10_test_loader.dataset.data[idx_tt],
                   title=f'test, L={label_tt}')

        def _show_contexts(idx_list, ax_offset):
            for j, idx_tr in enumerate(idx_list):
                label_tr = cifar10_train_loader.dataset.targets[idx_tr]
                infl = estimates['influence'][idx_tt, idx_tr]
                show_image(axs[i, j + ax_offset], cifar10_train_loader.dataset.data[idx_tr],
                           title=f'tr, L={label_tr}, infl={infl:.3f}')

        # show training examples with highest influence
        idx_sorted_tr = np.argsort(estimates['influence'][idx_tt])[::-1]
        _show_contexts(idx_sorted_tr[:n_context1], 1)

        # show random training examples from the same class
        idx_class = np.nonzero(np.array(cifar10_train_loader.dataset.targets) == label_tt)[0]
        idx_random = np.random.choice(idx_class, size=n_context2, replace=False)
        _show_contexts(idx_random, n_context1 + 1)

    plt.tight_layout()
    plt.savefig('results/cifar10-examples.pdf', bbox_inches='tight')

if __name__ == '__main__':

    args = argparser()
    rich_print(args)
    print ('\n')


    npz_fn = 'results/estimates_results.npz'
    if os.path.exists(npz_fn):
        estimates = np.load(npz_fn)
    else:
        metrics = estimate(args, logger=None)
        np.savez('results/estimates_results.npz', **metrics)
        
    #show_examples(estimates)
