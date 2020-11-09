from abc import abstractmethod
from functools import partial

import datasets
import pandas as pd
from fastai import imports
from fastai import learner
from fastai.callback import progress
from fastai.data import block, transforms
from blurr import utils

from blurr.modeling import summarization as model_sum
from blurr.data import question_answering as data_qa
from blurr.data import core as data_core
from blurr.modeling import core as model_core

from src.data_processing import data_processing
from src.augmentation import augmentations

# TODO
# adding custom metrics for qa
# implement augmenter from przemeks code
# implement nlp aug


def get_lr(lr, div1, div2):
    return slice(lr / div1, lr / div2)



class BlurrPipeline:
    def __init__(self, learn, parameters):
        self.learn = learn
        self.parameters = parameters

    def run(self):
        self.train()
        self.save_model()
        self.save_metrics()
        self.save_predictions()

    def train(self):
        params = self.parameters["train_params"][self.parameters["train_samples"]]
        unfrozen_epochs = params["epochs"][0]
        params["epochs"] = params["epochs"][1:]
        lr = self.learn.lr_find(suggestions=True).lr_min
        self.learn.fit_one_cycle(unfrozen_epochs, lr_max=lr)
        for epoch, unfreeze, lr_divs in zip(params.values()):
            if unfreeze == "all":
                self.learn.unfreeze()
            else:
                self.learn.freeze_to(unfreeze)
            self.learn.fit_one_cycle(epoch, lr_max=get_lr(lr, lr_divs[0],  lr_divs[1]))

    def save_model(self):
        self.learn.export(fname=self.parameters["model_save_paths"])

    def save_metrics(self):
        metrics = self.learn.csv_logger.read_log()
        final_metrics = metrics[:-1, :]
        final_metrics.to_csv(self.parameters["metrics_save_path"])

    def save_predictions(self):
        preds, target = self.parameters["predictions_save_paths"], self.parameters["targets_save_paths"]
        if not self.parameters["save_predictions"]:
            pass
        elif self.parameters["save_predictions"] == "train":
            self.learn.get_preds(ds_idx=1, with_loss=True, save_preds=preds, save_targs=target)
        elif self.parameters["save_predictions"] == "test":
            self.learn.get_preds(ds_idx=2, with_loss=True, save_preds=preds, save_targs=target)
        else:
            self.learn.get_preds(ds_idx=1, with_loss=True, save_preds=preds, save_targs=target)
            self.learn.get_preds(ds_idx=2, with_loss=True, save_preds=preds, save_targs=target)

    @classmethod
    def get_model_from_name(cls, pretrained_model_name, model_class):
        hf_arch, hf_config, hf_tokenizer, hf_model = utils.BLURR_MODEL_HELPER.get_hf_objects(
            pretrained_model_name,
            model_cls=model_class, # check if returns correct stuff without specifying a class
        )
        return hf_arch, hf_config, hf_tokenizer, hf_model

    @classmethod
    @abstractmethod
    def get_callbacks(cls, pre_config, config):
        pass

    @classmethod
    @abstractmethod
    def get_databunch_from_name(cls, ds, aug_fn, arch, tokenizer, params):
        pass

    @classmethod
    def get_augmentation_fn(cls, aug_name):
        if aug_name == "no_aug":
            return augmentations.NoAug
        else:
            raise ValueError(f"{aug_name} is not a supported augmentation mode")

    @classmethod
    def from_name(cls, exp_parameters):
        # will experiment abstraction be needed or exp_result class is enough
        # exp_parameters = add_env_vars(exp_parameters) # chyba blurr sam organie czy gpu etc.
        model_name, model_class = exp_parameters["pretrained_model_name"], exp_parameters["model_class"]
        aug_fn = cls.get_augmentation_fn(exp_parameters["augmentation"])
        arch, config, tokenizer, model = cls.get_model_from_name(model_name, model_class)
        databunch = cls.get_databunch_from_name(
            ds=exp_parameters["ds_name"],
            aug_fn=aug_fn,
            arch=arch,
            tokenizer=tokenizer,
            params=exp_parameters,
        )
        learn = cls.get_learner(
            arch=arch,
            pre_model=model,
            pre_config=config,
            config=exp_parameters,
            databunch=databunch
        )
        return cls(learn, exp_parameters)

    @classmethod
    def load_data(cls, dataset, train_samples):
        data_train = datasets.load_dataset(dataset, split="train")
        data_test = datasets.load_dataset(dataset, split="test")
        df_train = pd.DataFrame(data_train)
        df_test = pd.DataFrame(data_test)
        if train_samples:
            df_train = df_train[:train_samples]
            assert len(df_train) == train_samples, \
                f"Dataset is too small to return requested train sample count {train_samples}"
        df_train["is_valid"] = False
        df_test["is_valid"] = True
        df = pd.stack([df_train, df_test])
        return df

    @classmethod
    def add_custom_metrics(cls, learn, metrics):
        # custom_metrics = L([ValueMetric(partial(metric_value, metric_key=k), k) for k in metrics])
        # learn.metrics = learn.metrics + custom_metrics
        return learn

    @classmethod
    def get_learner(cls, databunch, arch, pre_model, pre_config, config):
        model = model_core.HF_BaseModelWrapper(pre_model)
        model_cb = cls.get_callbacks(pre_config, config["pre_config_overwrite"])
        model_cb = model_cb + [progress.CSVLogger]
        splitter = cls.get_splitter(arch)
        learn = learner.Learner(
            databunch,
            model,
            opt_func=config["opt_func"],
            loss_func=config["loss_func"],
            cbs=model_cb,
            splitter=splitter,
        )
        learn = cls.add_custom_metrics(learn, config["metrics"])
        learn.create_opt()
        learn.freeze()
        return learn

    @classmethod
    @abstractmethod
    def get_splitter(cls, arch, **kwargs):
        pass


class QuestionAnsweringPipeline(BlurrPipeline):
    @classmethod
    def get_splitter(cls, arch, **kwargs):
        return model_core.hf_splitter

    @classmethod
    def get_callbacks(cls, pre_config, config):
        return []

    @classmethod
    def get_databunch_from_name(cls, ds, aug_fn, arch, tokenizer, params):
        vocab = list(range(params["max_len"]))
        df = cls.load_data(ds, params["train_samples"])
        processed_df = data_processing.processing_from_name(df, ds, arch, tokenizer, params["max_len"])

        trunc_strat = 'only_second' if (tokenizer.padding_side == 'right') else 'only_first'
        hf_batch_tfm = data_qa.HF_QABatchTransform(arch, tokenizer,
                                           max_length=params["max_len"],
                                           truncation=trunc_strat,
                                           tok_kwargs={'return_special_tokens_mask': True})

        blocks = (
            data_core.HF_TextBlock(hf_batch_tfm=hf_batch_tfm),
            block.CategoryBlock(vocab=vocab),
            block.CategoryBlock(vocab=vocab)
        )

        def get_x(x):
            return (x.question, x.context) if (tokenizer.padding_side == 'right') else (x.context, x.question)

        # get_x=aug_fn(params["x_col"]),#transforms.ColReader

        dblock = block.DataBlock(blocks=blocks,
                           get_x=get_x,
                           get_y=[transforms.ColReader('tok_answer_start'), transforms.ColReader('tok_answer_end')],
                           splitter=transforms.ColSplitter(col="is_valid"),
                           n_inp=1)

        dls = dblock.dataloaders(processed_df, bs=params["bs"])
        return dls


class SummarizationPipeline(BlurrPipeline):
    @classmethod
    def get_splitter(cls, arch, **kwargs):
        return partial(model_sum.summarization_splitter, arch=arch)

    @classmethod
    def get_callbacks(cls, pre_config, config):
        text_gen_kwargs = {**pre_config.task_specific_params['summarization'], **config}
        model_cb = model_sum.HF_SummarizationModelCallback(text_gen_kwargs=text_gen_kwargs)
        return [model_cb]

    @classmethod
    def get_databunch_from_name(cls, ds, aug_fn, arch, tokenizer, params):
        # load data
        df = cls.load_data(ds, params["train_samples"])
        processed_df = data_processing.processing_from_name(df, ds, arch, tokenizer, params["max_len"])

        # convert to datablock (dataset specific fragment?)
        hf_batch_tfm = model_sum.HF_SummarizationBatchTransform(
            arch,
            tokenizer,
            max_length=params["max_len"]
        )
        blocks = (data_core.HF_TextBlock(hf_batch_tfm=hf_batch_tfm), imports.noop)

        # update ColReader with augmentation technique
        dblock = block.DataBlock(blocks=blocks,
                           get_x=aug_fn(params["x_col"]),#transforms.ColReader
                           get_y=transforms.ColReader(params["y_col"]),
                           splitter=transforms.ColSplitter(col="is_valid"))

        dls = dblock.dataloaders(processed_df, bs=params["bs"])
        return dls
