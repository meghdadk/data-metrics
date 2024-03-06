import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.resnet import ResNet101_Weights, ResNet18_Weights
from torchvision.models.vgg import VGG16_Weights
import torch.nn.init as init

_MODELS = {}


def init_params(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            init.kaiming_normal_(m.weight, mode='fan_in')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, std=1e-3)
            if m.bias is not None:
                init.constant_(m.bias, 0)

def _add_model(model_fn):
    _MODELS[model_fn.__name__] = model_fn
    return model_fn



@_add_model   
def resnet18(num_classes=10, **kwargs):
    weights = kwargs.get('weights', 'random')
    if weights == 'imagenet':
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    else:
        model = models.resnet18()

    # Adjust the final fully connected layer to num_classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    if weights == 'random':
        model.apply(init_params)

    return model


@_add_model  
def resnet101(num_classes=10, **kwargs):    
    weights = kwargs.get('weights', None)
    if weights is not None:
        model = models.resnet101(weights=ResNet101_Weights.DEFAULT)
    else:
        model = models.resnet101()

    # Adjust the final fully connected layer to num_classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    if weights is None:
        model.apply(init_params)

    return model

@_add_model  
def vgg16(num_classes=10, **kwargs):    
    weights = kwargs.get('weights', None)
    if weights is not None:
        model = models.vgg16(weights=VGG16_Weights.DEFAULT)
    else:
        model = models.vgg16()

    # Adjust the final fully connected layer to num_classes
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, num_classes)

    if weights is None:
        model.apply(init_params)

    return model


def load_model(model, seed, num_classes, **kwargs):

    model = _MODELS[model](num_classes, **kwargs)

    return model