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

def vggnet(n_class, model_name = 'vgg16', pretrained=False):
    if model_name == 'vgg11':
        model = models.vgg16(pretrained = pretrained)
    elif model_name == 'vgg11_bn':
        model = models.vgg16_bn(pretrained = pretrained)
    elif model_name == 'vgg13':
        model = models.vgg16(pretrained = pretrained)
    elif model_name == 'vgg13_bn':
        model = models.vgg16_bn(pretrained = pretrained)
    elif model_name == 'vgg16':
        model = models.vgg16_bn(pretrained = pretrained)
    elif model_name == 'vgg16_bn':
        model = models.vgg16(pretrained = pretrained)
    elif model_name == 'vgg19':
        model = models.vgg16_bn(pretrained = pretrained)
    elif model_name == 'vgg19_bn':
        model = models.vgg16_bn(pretrained = pretrained)

    if n_class !=1000: 
        model.classfier[6] = nn.Linear(model.classfier[6].in_features, n_class)
    
    return model


if __name__ == '__main__':
    model = vggnet(5, model_name = 'vgg16_bn', pretrained = True)



    print(model)

