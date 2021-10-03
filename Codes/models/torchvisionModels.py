import torchvision
import torchvision.models as models
import torch.nn as nn


def resnet(n_class, model_name='resnet18', pretrained=False):
    if model_name == 'resnet18':
        model = models.resnet18(pretrained)
    elif model_name == 'resnet34':
        model = models.resnet34(pretrained)
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained)
    elif model_name == 'resnet101':
        model = models.resnet101(pretrained)
    elif model_name == 'resnet152':
        model = models.resnet152(pretrained)
    else:
        raise ModuleNotFoundError

    if n_class != 1000: # imagenet
        model.fc = nn.Linear(model.fc.in_features, n_class)

    return model

