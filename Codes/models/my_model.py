import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import numpy as np


def get_last_layer_channel(model):
    _, c, _, _ = model(torch.ones((1,3,224,224))).shape
    return c

''' -------------------- Bottleneck block -------------------- '''
''' -------------------- Bottleneck block -------------------- '''
''' -------------------- Bottleneck block -------------------- '''
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(planes * 4)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

''' ----- Coronary Artery Calcium and Cardiovascular Risk Prediction Network ----- '''
''' ----- Coronary Artery Calcium and Cardiovascular Risk Prediction Network ----- '''
''' ----- Coronary Artery Calcium and Cardiovascular Risk Prediction Network ----- '''
class CACNet(nn.Module):
    def __init__(self, 
                n_class=5, 
                feature_extractor='resnet34', 
                feature_pretrained = True, 
                feature_freeze = True,
                block=Bottleneck, 
                layers_attention=[3, 4, 6, 3, 2]):
        super(CACNet, self).__init__()

        from . import get_feature_extractor

        self.inplanes = 64

        # feature extractor
        self.feature_extractor = \
            get_feature_extractor(model_name=feature_extractor, 
                                  pretrained=feature_pretrained, 
                                  feature_freeze=feature_freeze)
        # get channel size of the last layer in feature extractor                
        self.feat_outplanes = get_last_layer_channel(self.feature_extractor)
        
        # attention module
        self.attention_module = self.make_attention_module(block, layers_attention, self.feat_outplanes)
        # fully-connected layer
        self.fc = nn.Linear(self.feat_outplanes, n_class)

        # pooling layers 
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def make_attention_module(self, block, layers_attention, last_layer_channel=256): # layers_attention: [3, 4, 6, 3, 2]
        attention_layers = []

        attention_layers.append(nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)) 
        attention_layers.append(nn.BatchNorm2d(64))
        attention_layers.append(nn.ReLU(inplace=True))
        attention_layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        attention_layers.append(self._make_layer(block, 64, layers_attention[0])) 
        attention_layers.append(self._make_layer(block, 128, layers_attention[1], stride=2))
        attention_layers.append(self._make_layer(block, 256, layers_attention[2], stride=2))
        attention_layers.append(self._make_layer(block, 512, layers_attention[3], stride=2))
        attention_layers.append(self._make_layer(block, last_layer_channel // block.expansion, layers_attention[4], stride=1)) # additional to resnet50
        return nn.Sequential(*attention_layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        attn = self.attention_module(x)  
        feat = self.feature_extractor(x) 
        
        # attention applied feature map
        attn_feat = torch.mul(attn, feat)

        linear_in = self.avgpool(attn_feat).view(-1, self.feat_outplanes)
        pred = self.fc(linear_in)

        return pred


if __name__ == '__main__':
    import sys
    from os import path
    
    sys.path.append(path.dirname( path.dirname( path.abspath(__file__) ) ))
    from models import get_feature_extractor, SUPPORT_MODEL_LIST

    batch_size = 2
    n_class = 5
    for model_name in SUPPORT_MODEL_LIST:
        model = CACNet(n_class=n_class, feature_extractor=model_name)

        x = torch.ones((batch_size, 3, 224, 224))
        y = model(x)
        
        assert y.numel() == batch_size*n_class
        break
    
    print('All test passed.')