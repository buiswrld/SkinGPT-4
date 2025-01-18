import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .loss import get_loss_fn
from data.dataset import GeneralizedClassificationDataset
from .logger import TFLogger
from .evaluator import GeneralClassificationEvaluator
from skingpt4.models.detection import get_model
from .transformer import Transformer

class ClassificationTask(pl.LightningModule, TFLogger):
    """Standard interface for the trainer to interact with the model."""

    def __init__(self, params):
        super().__init__()
        self.save_hyperparameters(params)
        self.model = get_model(params)
        self.loss = get_loss_fn(params)
        self.evaluator = GeneralClassificationEvaluator()
        self.validation_outputs=[]
        self.lr = self.hparams.get('learning_rate', 5e-4)
        self.val_batches = self.hparams.get('val_batches', 1.0)
        self.oversample = self.hparams.get('oversample', False)
        self.oversample_factor = self.hparams.get('oversample_factor', 1.0) 
        self.oversample_col = self.hparams.get('oversample_col', 'label')
        self.downsample_factor = self.hparams.get('downsample_factor', 1.0)
        self.data_regime = self.hparams.get('data_regime', 1.0)
        self.dataset_path = self.hparams.get('dataset_path', "")
        self.classes = self.hparams.get('classes', ('Eczema', 'Allergic Contact Dermatitis','Urticaria', 'Psoriasis', 'Impetigo', 'Tinea'))
        
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
        self.evaluator.update((logits, y))
        self.validation_outputs.append(loss)
        return {'loss': loss}

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
        avg_loss = torch.stack(self.validation_outputs).mean()
        self.log("val_loss", avg_loss)
        metrics = self.evaluator.evaluate()
        self.evaluator.reset()
        self.log_dict(metrics)
        self.validation_outputs.clear()

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def on_test_epoch_end(self):
        return self.on_validation_epoch_end()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def train_dataloader(self):

        transformer = Transformer()
        transformer.downsample(self.downsample_factor)
        transformer.to_tensor()
        transformer.randomize_img(degree=1)
        transforms = transformer.get_transforms()
        dataset = GeneralizedClassificationDataset(
            dataset_path=self.dataset_path, 
            split="train", 
            transforms=transforms, 
            classes=self.classes, 
            data_regime=self.data_regime
        )
        if self.oversample:
            labels = dataset.dataset[self.oversample_col].tolist()
            class_counts = {cls: labels.count(cls) for cls in set(labels)}
            class_weights = {cls: (1.0 / count)*self.oversample_factor for cls, count in class_counts.items()}
            sample_weights = [class_weights[label] for label in labels]
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
            shuffle=False
        else:
            sampler = None
            shuffle = True
        print(f"Training set number of samples: {len(dataset)}")
        return DataLoader(dataset, shuffle=shuffle, sampler=sampler,
                          batch_size=2, num_workers=8)
 
    def val_dataloader(self):
        transformer = Transformer()
        transformer.to_tensor()
        transforms = transformer.get_transforms()
        dataset = GeneralizedClassificationDataset(
            dataset_path=self.dataset_path, 
            split="val", 
            transforms=transforms, 
            classes=self.classes,
        )
        print(f"Validation set number of samples: {len(dataset)}")
        return DataLoader(dataset, shuffle=False,
                          batch_size=self.val_batches, num_workers=8)

    def test_dataloader(self):
        transformer = Transformer()
        transformer.to_tensor()
        transforms = transformer.get_transforms()
        dataset = GeneralizedClassificationDataset(
            dataset_path=self.dataset_path, 
            split="test", 
            transforms=transforms, 
            classes=self.classes,
        )
        print(f"Testing set number of samples: {len(dataset)}")
        return DataLoader(dataset, shuffle=False,
                          batch_size=1, num_workers=8)