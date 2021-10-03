import torch
import torch.nn as nn

from models import get_model
from optimizer import get_optimizer
from dataset import get_dataloader

import argparse
import utils
import numpy as np
import random
import matplotlib

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, required=True)
    # dataset

    # learning hyperparameter
    parser.add_argument("--lr", type=int, default=1e-4, help='')
    parser.add_argument("--batch_size", type=int, default=32, help='')
    parser.add_argument("--epoch", type=int, default=100, help='')
    parser.add_argument("--eval_epoch", type=int, default=5, help='')
    # model hyperparameter
    parser.add_argument("--model_name", type=str, default='rexnetv1-1.0', choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'rexnetv1-1.0', 'rexnetv1-1.3', 'rexnetv1-1.5', 'rexnetv1-2.0', 'rexnetv1-3.0'])
    parser.add_argument("--pretrained", action='store_true', default=False, help='')
    parser.add_argument("--optimizer_name", type=str, default='Adam', choices=['Adam'])
    # etc
    parser.add_argument("--print_cycle", type=int, default=100, help='')
    parser.add_argument("--seed", type=int, default=42, help='')
    parser.add_argument("--n_class", type=int, default=5)

    args = parser.parse_args()
    return args

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def main():
    args = parse_args()
    set_seed(args.seed)


    assert torch.cuda.is_available()

    model = get_model(args)
    model.cuda()

    optimizer = get_optimizer(parameters=model.parameters(), lr=args.lr, optimizer_name=args.optimizer_name)

    train_dataloader = get_dataloader(batch_size = args.batch_size, split='train')
    val_dataloader = get_dataloader(batch_size = args.batch_size, split='val')
    test_dataloader = get_dataloader(batch_size = args.batch_size, split='test')

    criterion = nn.CrossEntropyLoss()

    save_manager = utils.SaveManager(args.version)

    print("Training starts")
    best_loss = 1e10
    for epoch in range(args.epoch):
        # train
        model.train()
        mean_loss = 0
        step = 0
        for batch in train_dataloader:
            x, y = batch
            x = x.cuda()
            y = y.cuda()

            out = model(x)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            mean_loss += loss.item()
            step+=1
            if step % args.print_cycle == 0 :
                print('Batch [{}/{}] Loss [{:.6f}]'.format(step, len(train_dataloader), mean_loss / step))
        print("Epoch [{}/{}] Training loss [{:.6f}]".format(epoch+1, args.epoch, mean_loss/step))
                
        
        # val
        if args.eval_epoch>0 and (epoch % args.eval_epoch == 0 or epoch+1 == args.epoch):
            preds = []
            labels = []
            with torch.no_grad():
                model.eval()
                mean_loss = 0
                step = 0
                for batch in val_dataloader:
                    x, y =batch
                    x = x.cuda()
                    y = y.cuda()

                    out = model(x)
                    loss = criterion(out, y)
                    mean_loss += loss.item()
                    step += 1
                    preds.append(out.data.cpu())
                    labels.append(y.data.cpu())
                preds = torch.cat(preds)
                labels = torch.cat(labels)
                acc = torch.mean((preds.argmax(1) == labels).float())

                mean_loss = mean_loss / step
                
                
                print("Epoch [{}/{}] Accuracy [{:.6f}] Validation loss [{:.6f}]".format(epoch + 1, args.epoch, acc, mean_loss))


                if mean_loss < best_loss:
                    best_loss = mean_loss
                    # model save
                    save_manager.save_checkpoint(model,optimizer,epoch)
                    print("======= Model Saved - Loss [{:.6f}] =======".format(best_loss))
    
    #test
    print(' ===================== End of Train ======================')
    
    model.load_state_dict(torch.load(save_manager.best_path)['model'])
    
    preds = []
    labels = []
    with torch.no_grad():
        model.eval()
        mean_loss = 0
        step = 0
        for (x,y) in test_dataloader:
            x = x.cuda()
            y = y.cuda()

            out = model(x)
            loss = criterion(out, y)
        
            preds.append(out.data.cpu())
            labels.append(y.data.cpu())
    preds = torch.cat(preds)
    labels = torch.cat(labels)
        
        
    acc = torch.mean((preds.argmax(1) == labels).float())
    print('Version {} - Best accuracy : {}'.format(args.version, acc))
    
    return



if __name__ == '__main__':
    main()