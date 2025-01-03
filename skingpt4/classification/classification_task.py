import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from .loss import get_loss_fn
from data.dataset import GeneralizedClassificationDataset
from .logger import TFLogger
from .evaluator import GeneralClassificationEvaluator
from skingpt4.models.detection import get_model

class ClassificationTask(pl.LightningModule, TFLogger):
    """Standard interface for the trainer to interact with the model."""

    def __init__(self, params):
        super().__init__()
        self.save_hyperparameters(params)
        self.model = get_model(params)
        self.loss = get_loss_fn(params)
        self.evaluator = GeneralClassificationEvaluator()
        self.validation_outputs=[]

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        """
        Returns:
            A dictionary of loss and metrics, with:
                loss(required): loss used to calculate the gradient
                log: metrics to be logged to the TensorBoard and metrics.csv
                progress_bar: metrics to be logged to the progress bar
                              and metrics.csv
        """
        x, y = batch["image"], batch["label"] 
        print(f"Training y shape: {y.shape}, y: {y.tolist()}")  # Debug print
        y= torch.argmax(y, dim=1)
        logits = self.forward(x)
        loss = self.loss(logits, y)
        self.log("loss", loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        x, y = batch['image'], batch['label']
        print(f"Validation y shape before update: {y.shape}, y: {y.tolist()}")  # Debug print
        y= torch.argmax(y, dim=1)
        logits = self.forward(x)
        loss = self.loss(logits, y)
        y_hat = (logits > 0).float()
        self.evaluator.update((logits, y))
        self.validation_outputs.append(loss) #############################
        return {'loss': loss}

    #def validation_epoch_end(self, outputs):
    def on_validation_epoch_end(self):
        """
        Aggregate and return the validation metrics

        Args:
        outputs: A list of dictionaries of metrics from `validation_step()'
        Returns: None
        Returns:
            A dictionary of loss and metrics, with:
                val_loss (required): validation_loss
                log: metrics to be logged to the TensorBoard and metrics.csv
                progress_bar: metrics to be logged to the progress bar
                              and metrics.csv
        """
        #avg_loss = torch.stack(outputs).mean()
        avg_loss = torch.stack(self.validation_outputs).mean()
        self.log("val_loss", avg_loss)
        metrics = self.evaluator.evaluate()
        self.evaluator.reset()
        self.log_dict(metrics)
        self.validation_outputs.clear()

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    #def test_epoch_end(self, outputs):
    def on_test_epoch_end(self):
        #return self.validation_epoch_end(outputs)
        return self.on_validation_epoch_end()

    def configure_optimizers(self):
        lr = self.hparams.get('learning_rate', 5e-4)
        return torch.optim.Adam(self.parameters(), lr=lr)
    
    def train_dataloader(self):
        '''
        dataset_path = self.hparams.get('dataset_path', "")
        transforms_list = [ transforms.Resize((810, 1080)),
                            transforms.ToTensor(), #(C, H, W) from (H, W, C) 
                            transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomVerticalFlip(0.5),
                          ]
        dataset = GeneralizedClassificationDataset(dataset_path=dataset_path, split="train", transforms=transforms.Compose(transforms_list), classes=self.hparams.get('classes'))
        print(f"Training set number of samples: {len(dataset)}")
        return DataLoader(dataset, shuffle=True,
                          batch_size=2, num_workers=8)
        '''
        dataset_path = self.hparams.get('dataset_path', "")
        transforms_list = [
            transforms.Resize((810, 1080)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
        ]
        dataset = GeneralizedClassificationDataset(dataset_path=dataset_path, split="train", transforms=transforms.Compose(transforms_list), classes=self.hparams.get('classes'))
        print(f"Training set number of samples: {len(dataset)}")
        dataloader = DataLoader(dataset, shuffle=True, batch_size=2, num_workers=8)
        for batch in dataloader:
            images, labels = batch["image"], batch["label"]
            print(f"Train DataLoader - images shape: {images.shape}, labels shape: {labels.shape}, labels: {labels.tolist()}")
            break
        return dataloader
 
    def val_dataloader(self):
        dataset_path = self.hparams.get('dataset_path', "")
        transforms_list = [ transforms.Resize((810, 1080)),transforms.ToTensor()]
        dataset = GeneralizedClassificationDataset(dataset_path=dataset_path, split="val", transforms=transforms.Compose(transforms_list), classes=self.hparams.get('classes'))
        print(f"Validation set number of samples: {len(dataset)}")
        return DataLoader(dataset, shuffle=False,
                          batch_size=1, num_workers=8)

    def test_dataloader(self):
        dataset_path = self.hparams.get('dataset_path', "")
        transforms_list = [ transforms.Resize((810, 1080)),transforms.ToTensor()]
        dataset = GeneralizedClassificationDataset(dataset_path=dataset_path, split="test", transforms=transforms.Compose(transforms_list), classes=self.hparams.get('classes'))
        print(f"Testing set number of samples: {len(dataset)}")
        return DataLoader(dataset, shuffle=False,
                          batch_size=1, num_workers=8)