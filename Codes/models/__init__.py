import torch
from models.rexnetv1 import ReXNetV1
from models.torchvisionModels import resnet
from models.torchvisionModels import vggnet

#! attention model
from models.my_model import CACNet


from config import *


SUPPORT_MODEL_LIST = [
    'resnet18',
    'resnet34',
    'resnet50',
    'resnet101',
    'resnet152',
    'vgg11',
    'vgg11_bn',
    'vgg13',
    'vgg13_bn',
    'vgg16',
    'vgg16_bn',
    'vgg19',
    'vgg19_bn',
    'CACNet'
]

def _test_model(model, num_class, input_size):
    dummy_input = torch.randn(1, 3, input_size, input_size)
    pred = model(dummy_input)

    print(pred.shape)

    return True


def get_model(args, verbose = False):
    model_name = MODEL_NAME

    if 'resnet' in model_name:
        return resnet(n_class=args.n_class, model_name=model_name, pretrained=args.pretrained)
    elif 'rexnetv1' in model_name:
        multiplier = float(model_name.split('-')[1])
        model = ReXNetV1(width_mult=multiplier)
        if args.pretrained:
            model.load_state_dict(torch.load('./pretrained/rexnetv1_{:.1f}.pth'.format(multiplier)))
        
    elif model_name == 'CACNet':
        model = CACNet (
            n_class = NUM_CLASS,
            feature_extractor = FEATURE_EXTRACTOR,
            feature_pretrained = FEATURE_PRETRAINED,
            feature_freeze = FEATURE_FREEZE
        )

        # TODO
        if PRETRAINED:
            raise NotImplementedError

    else:
        raise ValueError(f'Supported Molers are as follows: {SUPPORT_MODEL_LIST}')

    if verbose:
        if _test_model(model, NUM_CLASS, INPUT_RESOLUTION):
            pass


    return model    

def get_feature_extractor(model_name:str, pretrained:bool, feature_freeze:bool):
    
    dummy_value = 1000

    if 'resnet' in model_name:
        model = resnet(dummy_value, model_name, pretrained)
        feature_extractor = torch.nn.Sequential(*list(model.children())[:-2])

    elif 'vgg' in model_name:
        model = vggnet(dummy_value, model_name, pretrained)  
        feature_extractor = model.features

    else:
        raise ModuleNotFoundError

    if feature_freeze == True:
        for param in feature_extractor.parameters():
            param.requires_grad_(False)

    return feature_extractor

# export CUDA_VISIBLE_DEVICES=0,1  # (use both gpus)


if __name__ == '__main__':
    vgg_feature_extractor1 = get_feature_extractor(model_name = 'vgg19', pretrained = True, feature_freeze= True)

    dummy_input1 = torch.ones((68,3,224,224)) / 2.


    vgg_feature_extractor1 = vgg_feature_extractor1.cuda()

    for _ in range(100):
        dummy_input1 = dummy_input1.cuda()
        out1 = vgg_feature_extractor1(dummy_input1)

        print (out1.shape)


    # print (vgg_feature_extractor1)