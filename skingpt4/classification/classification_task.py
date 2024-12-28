import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from .loss import get_loss_fn
from data.dataset import GeneralizedClassificationDataset
from .logger import TFLogger
from .evaluator import GeneralClassificationEvaluator
from skingpt4.models.skin_gpt4 import skingpt4

#TODO ~ view Blip2Base (blip2.py) & BaseModel (base_model.py)
class ClassificationTask(pl.LightningModule, TFLogger):
    """Standard interface for the trainer to interact with the model."""

    def __init__(self, params):
        super().__init__()
        self.save_hyperparameters(params)
        self.model = skingpt4.from_config(params)
        self.loss = get_loss_fn(params)
        self.evaluator = GeneralClassificationEvaluator()

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
        #TODO ~ examine bugfix effects
        x, y = batch, batch["label"] 
        logits = self.forward(x)
        loss = self.loss(logits, y)
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_nb):
        #TODO ~ examine bugfix effects
        x, y = batch, batch['label']
        logits = self.forward(x)
        loss = self.loss(logits.view(-1), y)
        y_hat = (logits > 0).float()
        self.evaluator.update((torch.sigmoid(logits), y))
        return loss

    def validation_epoch_end(self, outputs):
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
        avg_loss = torch.stack(outputs).mean()
        self.log("val_loss", avg_loss)
        metrics = self.evaluator.evaluate()
        self.evaluator.reset()
        self.log_dict(metrics)

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)

    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=0.02)]
    
    #TODO ~ Address GeneralizedClassificationDataset 
    def train_dataloader(self):
        dataset_path = self.hparams.get('dataset_path', "")
        transforms_list = [ transforms.ToTensor(), #(C, H, W) from (H, W, C) 
                            transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomVerticalFlip(0.5),
                            transforms.RandomAffine(90),
                          ]
        dataset = GeneralizedClassificationDataset(dataset_path=dataset_path, split="valid", transforms=transforms.Compose(transforms_list))
        return DataLoader(dataset, shuffle=True,
                          batch_size=2, num_workers=8)

    def val_dataloader(self):
        dataset_path = self.hparams.get('dataset_path', "")
        transforms_list = [ transforms.ToTensor()]
        dataset = GeneralizedClassificationDataset(dataset_path=dataset_path, split="valid", transforms=transforms.Compose(transforms_list))
        return DataLoader(dataset, shuffle=False,
                          batch_size=1, num_workers=8)

    def test_dataloader(self):
        dataset_path = self.hparams.get('dataset_path', "")
        transforms_list = [ transforms.ToTensor()]
        dataset = GeneralizedClassificationDataset(dataset_path=dataset_path, split="valid", transforms=transforms.Compose(transforms_list))
        return DataLoader(dataset, shuffle=False,
                          batch_size=1, num_workers=8)