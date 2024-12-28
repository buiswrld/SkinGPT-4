import torch.distributed as dist
import numpy as np
import ignite

#from . import segmentation
from . import classification

#TODO ~ Add SegmentationEvaluator and Seg utility folder (similar to classification)
#TODO ~ Replace with general cross entropy

class BinaryClassificationEvaluator(
        ignite.metrics.EpochMetric):
    def __init__(self, threshold=None, check_compute_fn=True):
        self._threshold = threshold
        super().__init__(self.compute_fn)

    def compute_fn(self, prob, y):
        return classification.get_binary_metrics(
            prob, y, threshold=self._threshold)

    def evaluate(self):
        return self.compute()