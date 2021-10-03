import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
from PIL import Image
import pydicom
import os


def get_dataloader(batch_size, split='train', num_workers=8):
    if split == 'train':
        train_path = '../Data/train_dataset.parquet'
        train_df = pd.read_parquet(train_path, engine='pyarrow')
        train_dataset = CoronaryArteryDataset(train_df, is_training=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False)
        return train_loader
    elif split == 'val':
        val_path = '../Data/val_dataset.parquet'
        val_df = pd.read_parquet(val_path, engine='pyarrow')
        val_dataset = CoronaryArteryDataset(val_df)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
        return val_loader
    else: # test
        val_path = '../Data/test_dataset.parquet'
        val_df = pd.read_parquet(val_path, engine='pyarrow')
        val_dataset = CoronaryArteryDataset(val_df)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
        return val_loader

class CoronaryArteryDataset(Dataset):
    def __init__(self, df, is_training=False):
        self.is_training = is_training
        self.data = df.values
        self.init_transforms()

    def init_transforms(self,):
        if self.is_training:
            self.transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomChoice([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomAffine(
                        degrees=15, translate=(0.2, 0.2),
                        scale=(0.8, 1.2), shear=15, resample=Image.BILINEAR)
                ]),
                transforms.ToTensor(),
                ])
        else:
            # val, test
            self.transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
        return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        dcm_path, pat_id, study_date, score = self.data[index]
        
        array_path = os.path.join('../Data/resized_224/',dcm_path.replace('.dcm','.npy'))
        image = np.load(array_path)

#         dcm = pydicom.read_file(os.path.join('../Data/',dcm_path))
#         image = dcm.pixel_array
        image = image - image.min()
        image = image / image.max() * 255
        image = Image.fromarray(image.astype('uint8')).convert("RGB")
        image = self.transforms(image)
        score = torch.LongTensor([score]).squeeze()

        return image, score