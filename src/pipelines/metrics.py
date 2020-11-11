from fastai.callback import core as callback_core
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class TextF1(callback_core.Callback):
    def on_epoch_begin(self, **kwargs):
        self.sum_f1, self.total = 0, 0

    def on_batch_end(self, last_output, last_target, **kwargs):
        import pdb;
        pdb.set_trace()
        self.total += 1
        common_tokens = pred_tokens = truth_tokens = 1
        prec = len(common_tokens) / len(pred_tokens)
        rec = len(common_tokens) / len(truth_tokens)
        self.sum_f1 += 2 * (prec * rec) / (prec + rec)

    def on_epoch_end(self, **kwargs):
        self.metric = self.sum_f1 / self.total


class ExactMatch(callback_core.Callback):
    def on_epoch_begin(self, **kwargs):
        self.correct, self.total = 0, 0

    def on_batch_end(self, last_output, last_target, **kwargs):
        import pdb;
        pdb.set_trace()
        preds = last_output.argmax(1)
        self.correct += sum(preds == last_target)
        self.total += preds.shape[0]

    def on_epoch_end(self, **kwargs):
        self.metric = self.correct / self.total


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
