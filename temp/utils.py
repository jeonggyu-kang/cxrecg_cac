import torch
import os
import matplotlib.pyplot as plt

class SaveManager():
    def __init__(self, version=None, save_root='../save/', load=False):
        assert version or load
        self.best_path = None
        # make save dir
        if not os.path.exists(save_root):
            os.makedirs(save_root)


        if load:
            raise NotImplementedError
        else:
            # new experiment
            exp_num = self.get_exp_num(save_root)
            self.save_dir = os.path.join(save_root, '{:04d}_{}'.format(exp_num, version))

    def log(self, text):

        return


    def get_exp_num(self, save_root):
        previous_exps = os.listdir(save_root)
        if len(previous_exps)==0:
            return 1
        else:
            return max([int(i.split('_')[0]) for i in os.listdir(save_root)]) + 1

    def save_checkpoint(self, model, optimizer, epoch):
        save_dir = os.path.join(self.save_dir, 'model')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)


        save_dict = {'model':model.state_dict(),
                     'optimizer':optimizer.optimizer.state_dict(),
                     'epoch':epoch}
    
        self.best_path = os.path.join(save_dir,'model.pth')
        torch.save(save_dict,self.best_path)
        return