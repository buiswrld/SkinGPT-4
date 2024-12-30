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
        logits = self.forward(x)
        loss = self.loss(logits, y)
        self.log("loss", loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        x, y = batch['image'], batch['label']
        logits = self.forward(x)
        loss = self.loss(logits, y)
        y_hat = (logits > 0).float()
        self.evaluator.update((torch.sigmoid(logits), y))
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
        avg_loss = torch.stack(self.outputs).mean()
        self.log("val_loss", avg_loss)
        metrics = self.evaluator.evaluate()
        self.evaluator.reset()
        self.log_dict(metrics)
        print(metrics)

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    #def test_epoch_end(self, outputs):
    def on_test_epoch_end(self):
        #return self.validation_epoch_end(outputs)
        return self.on_validation_epoch_end()

    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=0.02)]
    
    def train_dataloader(self):
        dataset_path = self.hparams.get('dataset_path', "")
        transforms_list = [ transforms.ToTensor(), #(C, H, W) from (H, W, C) 
                            transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomVerticalFlip(0.5),
                            transforms.RandomAffine(90),
                          ]
        dataset = GeneralizedClassificationDataset(dataset_path=dataset_path, split="train", transforms=transforms.Compose(transforms_list))
        print(f"Training set number of samples: {len(dataset)}")
        return DataLoader(dataset, shuffle=True,
                          batch_size=2, num_workers=8)

    def val_dataloader(self):
        dataset_path = self.hparams.get('dataset_path', "")
        transforms_list = [ transforms.ToTensor()]
        dataset = GeneralizedClassificationDataset(dataset_path=dataset_path, split="valid", transforms=transforms.Compose(transforms_list))
        print(f"Validation set number of samples: {len(dataset)}")
        return DataLoader(dataset, shuffle=False,
                          batch_size=1, num_workers=8)

    def test_dataloader(self):
        dataset_path = self.hparams.get('dataset_path', "")
        transforms_list = [ transforms.ToTensor()]
        dataset = GeneralizedClassificationDataset(dataset_path=dataset_path, split="test", transforms=transforms.Compose(transforms_list))
        print(f"Testing set number of samples: {len(dataset)}")
        return DataLoader(dataset, shuffle=False,
                          batch_size=1, num_workers=8)