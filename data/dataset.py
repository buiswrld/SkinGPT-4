import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
import pandas as pd 


class GeneralizedClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, split, transforms=None):
        df = pd.read_csv(dataset_path)
        self.dataset = df.loc[df['split'] == split].reset_index(drop=True) 
        self.transforms = transforms 

    def __len__(self):
        return len(self.dataset) 
    
    #TODO: ~ Address the return_dict to accomodate SCIN
    def __getitem__(self, index):
        sample = self.dataset.loc[index] 
        return_dict = {} 
        return_dict["Google_image"] = Image.open(sample["Google_image_path"]) #(H, W, C)
        if self.transforms:
            return_dict["Google_image"] = self.transforms(return_dict["Google_image"])
        return_dict['label_classification'] = sample['label_classification']
        return return_dict 


class ImageClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, image_path=None, labels=None, transforms=None):
        self._image_path = image_path
        self._labels = labels
        self._transforms = transforms

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, index):
        label = torch.tensor(np.float64(self._labels[index]))
        image = Image.open(self._image_path[index]).convert('RGB')
        if self._transforms is not None:
            image = self._transforms(image)

        return image, label