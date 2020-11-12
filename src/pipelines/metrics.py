import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def calculate_text_f1(pred_starts, pred_ends, target_starts, target_ends):
    pred_starts = torch.clamp(pred_starts, min=0)
    pred_ends = torch.max(pred_starts, pred_ends)

    prediction_lenght = pred_ends - pred_starts
    target_lenght = target_ends - target_starts

    pred_zero_mask = prediction_lenght == 0
    target_zero_mask = target_lenght == 0
    correct_zeros = pred_zero_mask & target_zero_mask

    common_tokens = torch.clamp(torch.min(pred_ends, target_ends) - torch.max(pred_starts, target_starts), min=0)
    prec = common_tokens / prediction_lenght
    rec = common_tokens / target_lenght
    f1_score = 2 * (prec * rec) / (prec + rec)

    f1_score[pred_zero_mask] = 0
    f1_score[target_zero_mask] = 0
    f1_score[torch.isnan(f1_score)] = 0
    f1_score[correct_zeros] = 1

    f1_score = f1_score.mean()

    return f1_score


def calculate_text_exact_match(pred_starts, pred_ends, target_starts, target_ends):
    exact_match = np.true_divide(((pred_starts == target_starts) & (pred_ends == target_ends)).sum(), len(pred_starts))
    return exact_match


def calculate_qa_metric(pred_starts, pred_ends, target_starts, target_ends, metric_key):
    if (metric_key == 'exact_match'): return calculate_text_exact_match(pred_starts, pred_ends, target_starts,
                                                                        target_ends)
    if (metric_key == 'f1'): return calculate_text_f1(pred_starts, pred_ends, target_starts, target_ends)


def compute_binary_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def compute_multiclass_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, micro_f1, _ = precision_recall_fscore_support(labels, preds, average='micro', zero_division=0)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'micro_precision': precision,
        'micro_recall': recall,
        'micro_f1': micro_f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1
    }
