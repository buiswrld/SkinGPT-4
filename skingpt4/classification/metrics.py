# FILE: metrics.py

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
)

def get_optimal_f1(groundtruth, probabilities, return_threshold=False):
    """Get threshold maximizing f1 score."""
    prec, rec, threshold = precision_recall_curve(groundtruth, probabilities)
    f1_values = 2 * (prec * rec) / (prec + rec)
    argmax_f1 = np.nanargmax(f1_values)
    max_f1 = np.nanmax(f1_values)
    if return_threshold:
        return max_f1, threshold[argmax_f1]
    else:
        return max_f1

def get_max_precision_above_recall(groundtruth, probabilities, value, return_threshold=False):
    """Get maximum precision such that recall >= value."""
    if value > 1:
        raise ValueError(f"Cannot attain a recall of {value}")
    prec, rec, threshold = precision_recall_curve(groundtruth, probabilities)
    max_prec_above_rec = max(p for p, r in zip(prec, rec) if r >= value)
    if return_threshold:
        index = list(prec).index(max_prec_above_rec)
        return max_prec_above_rec, threshold[index - 1]
    else:
        return max_prec_above_rec

def get_multiclass_metrics(probs, labels):
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    if isinstance(probs, torch.Tensor):
        probs = probs.cpu().numpy()
    
    
    preds = np.argmax(probs, axis=1)
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average='weighted')
    rec = recall_score(labels, preds, average='weighted')
    f1 = f1_score(labels, preds, average='weighted')
    #auprc = average_precision_score(labels, probs, average='weighted')
    #auroc = roc_auc_score(labels, probs, average='weighted', multi_class='ovr')
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        #'auprc': auprc,
        #'auroc': auroc,
    }