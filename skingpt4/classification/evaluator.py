import torch.distributed as dist
import numpy as np
import ignite

from .metrics import get_multiclass_metrics

class GeneralClassificationEvaluator(
        ignite.metrics.EpochMetric):
    def __init__(self, check_compute_fn=True):
        super().__init__(self.compute_fn)

    def compute_fn(self, prob, y):
        return get_multiclass_metrics(prob, y)

    def evaluate(self):
        return self.compute()