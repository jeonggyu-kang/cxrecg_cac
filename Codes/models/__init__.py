import torch
from models.rexnetv1 import ReXNetV1
from models.torchvisionModels import resnet

def get_model(args):
    model_name = args.model_name

    if 'resnet' in model_name:
        return resnet(n_class=args.n_class, model_name=model_name, pretrained=args.pretrained)
    elif 'rexnetv1' in model_name:
        multiplier = float(model_name.split('-')[1])
        model = ReXNetV1(width_mult=multiplier)
        if args.pretrained:
            model.load_state_dict(torch.load('./pretrained/rexnetv1_{:.1f}.pth'.format(multiplier)))
        return model