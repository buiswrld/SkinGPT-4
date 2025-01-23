import torch.distributed as dist
import numpy as np
import ignite
import torch.nn.functional as F

from .metrics import get_multiclass_metrics

class GeneralClassificationEvaluator(
        ignite.metrics.EpochMetric):
    def __init__(self, check_compute_fn=True):
        super().__init__(self.compute_fn)

    def compute_fn(self, logits, y):
        prob = F.softmax(logits, dim=1)
        return get_multiclass_metrics(prob, y, num_classes=logits.shape[1])

    def evaluate(self):
        return self.compute()