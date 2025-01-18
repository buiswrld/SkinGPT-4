import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
import pandas as pd 


class GeneralizedClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, split, transforms=None, classes=('Eczema', 'Allergic Contact Dermatitis','Urticaria', 'Psoriasis', 'Impetigo', 'Tinea'), data_regime=1.0):
        print(classes)
        df = pd.read_csv(dataset_path)
        self.dataset = df.loc[df['split'] == split].reset_index(drop=True) 
        self.transforms = transforms 
        self.data_regime = data_regime
        if isinstance(classes, str):
            self.class_names = classes.split(',')
        elif isinstance(classes, tuple):
            self.class_names = list(classes)
        else:
            raise ValueError("classes input is neither a comma-separated string nor a tuple")
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.class_names)}
        
        #if not self.data_regime == 1.0:
            #self._apply_data_regime()

    def __len__(self):
        return len(self.dataset) 
    
    def __getitem__(self, index):
        sample = self.dataset.loc[index]
        image = Image.open(sample["image_path"]).convert('RGB')
        label = self.class_to_idx[sample["label"]]

        if self.transforms:
            image = self.transforms(image)

        return {"image": image, "label": label}

    def _apply_data_regime(self):
        if self.data_regime > 1.0 or self.data_regime <= 0.0:
            raise ValueError("data_regime must be in range (0, 1]")
        num_samples = int(len(self.dataset) * self.data_regime)
        indices = np.random.choice(len(self.dataset), num_samples, replace=False)
        self.dataset = self.dataset.iloc[indices].reset_index(drop=True)


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