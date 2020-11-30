"""
Simple training loop; Based on https://github.com/karpathy/minGPT/blob/master/mingpt/trainer.py
"""

import math
import logging

from tqdm.auto import tqdm
import numpy as np

import torch
from torch.utils.data.dataloader import DataLoader

logger = logging.getLogger(__name__)


class Prediction:
    def __init__(self, label_ids, predictions):
        self.label_ids = label_ids
        self.predictions = predictions


class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 0.001
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.01  # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6  # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9  # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0  # for DataLoader
    cuda = True

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Trainer:

    def __init__(self, model, train_dataset, test_dataset, collator, config, compute_metrics):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.collator = collator
        self.config = config
        self.compute_metrics = compute_metrics

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if config.cuda and torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def save_checkpoint(self, valid_loss=None):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        log_str = f"saving {self.config.ckpt_path}"
        if valid_loss:
            log_str += f" valid loss: {valid_loss}"
        logger.info(log_str)
        torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = torch.optim.AdamW(raw_model.parameters(), lr=config.learning_rate, betas=config.betas)

        def run_epoch(split):
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers,
                                collate_fn=self.collator)

            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            for it, (x, y) in pbar:

                # place data on the correct device
                x = x.to(self.device)
                y = y.to(self.device)

                # forward the model
                with torch.set_grad_enabled(is_train):
                    logits, loss = model(x, y)
                    loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus
                    losses.append(loss.item())

                if is_train:

                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()

                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (y >= 0).sum()  # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    # report progress
                    pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")

            if not is_train:
                valid_loss = float(np.mean(losses))
                print(f"valid loss: {valid_loss}")
                return valid_loss

        best_loss = float('inf')
        self.tokens = 0  # counter used for learning rate decay
        for epoch in range(config.max_epochs):

            run_epoch('train')
            if self.test_dataset is not None:
                valid_loss = run_epoch('test')

            # supports early stopping based on the test loss, or just save always if no test set is provided
            good_model = self.test_dataset is None or valid_loss < best_loss
            if self.config.ckpt_path is not None and good_model:
                best_loss = valid_loss
                self.save_checkpoint()

    def evaluate(self, data):
        model, config = self.model.eval(), self.config
        loader = DataLoader(data, shuffle=True, pin_memory=True,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers,
                            collate_fn=self.collator)
        outputs = list()
        labels_list = list()
        with torch.no_grad():
            for x, y in loader:
                out, _ = model(x)
                outputs.append(out)
                labels_list.append(y)
        predictions = torch.cat(outputs)
        label_ids = torch.cat(labels_list)
        pred = Prediction(label_ids, predictions)
        return self.compute_metrics(pred)
