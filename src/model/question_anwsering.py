from functools import partial

import torch
from blurr.modeling import core as blurr_core
from fastai import learner
from fastcore import basics, foundation

from src.pipelines import metrics


class HF_QstAndAnsModelCallbackWithMetrics(blurr_core.HF_BaseModelCallback):
    def __init__(self, tok_metrics=("exact_match", "f1"), **kwargs):
        self.run_before = learner.Recorder

        basics.store_attr(self=self, names='tok_metrics, kwargs')
        self.custom_metrics_dict = {k: None for k in tok_metrics}

        self.do_setup = True

    def after_pred(self):
        super().after_pred()
        self.learn.pred = (self.pred.start_logits, self.pred.end_logits)

    def setup(self):
        if (not self.do_setup): return
        custom_metric_keys = self.custom_metrics_dict.keys()
        custom_metrics = foundation.L([learner.ValueMetric(partial(self.metric_value, metric_key=k), k)
                                       for k in custom_metric_keys])
        self.learn.metrics = self.learn.metrics + custom_metrics
        self.do_setup = False

    def before_fit(self):
        self.setup()

    def after_batch(self):
        if (self.training or self.learn.y is None): return

        pred_starts, pred_ends = self.learn.pred[0].argmax(dim=1), self.learn.pred[1].argmax(dim=1)
        target_starts, target_ends = self.learn.yb

        self.preds_start = torch.cat([self.preds_start, pred_starts.cpu()])
        self.preds_end = torch.cat([self.preds_end, pred_ends.cpu()])
        self.targets_start = torch.cat([self.targets_start, target_starts.cpu()])
        self.targets_end = torch.cat([self.targets_end, target_ends.cpu()])

    def before_validate(self):
        self.preds_start, self.preds_end = torch.Tensor([]), torch.Tensor([])
        self.targets_start, self.targets_end = torch.Tensor([]), torch.Tensor([])

    def after_validate(self):
        if (len(self.preds_start) < 1): return

        for k in self.custom_metrics_dict.keys():
            self.custom_metrics_dict[k] = metrics.calculate_qa_metric(
                self.preds_start, self.preds_end, self.targets_start, self.targets_end, metric_key=k)

    def metric_value(self, metric_key):
        return self.custom_metrics_dict[metric_key]
