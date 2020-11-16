from abc import abstractmethod
from functools import partial

import datasets
import torch
import pandas as pd
from fastai import imports
from fastai import learner
from fastai.callback import progress
from fastai.data import block, transforms
from blurr import utils
from blurr.modeling import summarization as model_sum
from blurr.modeling import question_answering as model_qa
from blurr.data import question_answering as data_qa
from blurr.data import core as data_core
from blurr.modeling import core as model_core

from src.data_processing import data_processing
from src.model import question_anwsering
from src.augmentation import augmentations
from src import training_utils


class BlurrPipeline:
    def __init__(self, learn, parameters):
        self.learn = learn
        self.parameters = parameters

    def run(self):
        self.train()
        self.save_metrics()
        self.save_predictions()
        self.save_model()

    def train(self):
        print("training started")
        params = self.parameters["train_params"][self.parameters["train_samples"]]
        unfrozen_epochs = params["epochs"][0]
        params["epochs"] = params["epochs"][1:]

        lr = self.learn.lr_find().lr_min
        self.learn.fit_one_cycle(unfrozen_epochs, lr_max=lr)
        for epoch, unfreeze, lr_divs in zip(*params.values()):
            if unfreeze == "all":
                self.learn.unfreeze()
            else:
                self.learn.freeze_to(unfreeze)
            self.learn.fit_one_cycle(epoch, lr_max=training_utils.get_lr(lr, lr_divs[0],  lr_divs[1]))

    def save_model(self):
        print("saving model started")
        self.learn.remove_cb(progress.CSVLogger)
        self.learn.save(self.parameters["model_save_paths"])

    def save_metrics(self):
        print("saving metrics started")
        metrics = self.learn.csv_logger.read_log()
        final_metrics = metrics#.iloc[-1:, :]
        final_metrics.to_csv(self.parameters["metrics_save_paths"])

    def save_predictions(self):
        tokenizer = self.learn.dls.before_batch[0].hf_tokenizer
        if not self.parameters["save_predictions"]:
            pass
        elif self.parameters["save_predictions"] == "train":
            print("saving train predictions started")
            preds = self.get_predictions(ds_type="train", tokenizer=tokenizer)
            preds.to_csv(self.parameters["predictions_save_paths"])
        elif self.parameters["save_predictions"] == "test":
            print("saving test predictions started")
            preds = self.get_predictions(ds_type="valid", tokenizer=tokenizer)
            preds.to_csv(self.parameters["targets_save_paths"])
        else:
            print("saving all predictions started")
            preds = self.get_predictions(ds_type="train", tokenizer=tokenizer)
            preds.to_csv(self.parameters["predictions_save_paths"])
            preds = self.get_predictions(ds_type="valid", tokenizer=tokenizer)
            preds.to_csv(self.parameters["targets_save_paths"])

    @abstractmethod
    def get_batch_predictions(self, ds_type, tokenizer):
        pass

    @classmethod
    @abstractmethod
    def get_callbacks(cls, pre_config, config):
        pass

    @classmethod
    @abstractmethod
    def get_databunch_from_name(cls, ds, aug_fn, arch, tokenizer, params):
        pass

    @classmethod
    @abstractmethod
    def get_splitter(cls, arch, **kwargs):
        pass

    @classmethod
    def from_name(cls, exp_parameters):
        model_name, model_class = exp_parameters["pretrained_model_name"], exp_parameters["model_class"]
        print(f"create pipeline from name")
        aug_fn = cls.get_augmentation_fn(exp_parameters["augmentation"])
        print(f"augmentation {exp_parameters['augmentation']} loaded")
        arch, config, tokenizer, model = cls.get_model_from_name(model_name, model_class)
        print(f"model {model_name} loaded")
        databunch = cls.get_databunch_from_name(
            ds=exp_parameters["ds_name"],
            aug_fn=aug_fn,
            arch=arch,
            tokenizer=tokenizer,
            params=exp_parameters,
        )
        print(f"dataset {exp_parameters['ds_name']} loaded")
        learn = cls.get_learner(
            arch=arch,
            pre_model=model,
            pre_config=config,
            config=exp_parameters,
            databunch=databunch
        )
        print(f"learner setup finished")
        return cls(learn, exp_parameters)

    @classmethod
    def get_model_from_name(cls, pretrained_model_name, model_class):
        hf_arch, hf_config, hf_tokenizer, hf_model = utils.BLURR_MODEL_HELPER.get_hf_objects(
            pretrained_model_name,
            model_cls=model_class,
        )
        return hf_arch, hf_config, hf_tokenizer, hf_model

    @classmethod
    def load_data(cls, dataset, train_samples):
        data_train = datasets.load_dataset(*dataset, split="train")
        data_test = datasets.load_dataset(*dataset, split="validation")
        df_train = pd.DataFrame(data_train)
        df_test = pd.DataFrame(data_test)
        if train_samples != "all":
            df_train = df_train[:train_samples]
            assert len(df_train) == train_samples, \
                f"Dataset is too small to return requested train sample count {train_samples}"
        df_train["is_valid"] = False
        df_test["is_valid"] = True
        df = pd.concat([df_train, df_test])
        return df

    @classmethod
    def get_learner(cls, databunch, arch, pre_model, pre_config, config):
        model = model_core.HF_BaseModelWrapper(pre_model)
        model_cb = cls.get_callbacks(pre_config, config["pre_config_overwrite"])
        model_cb += [progress.CSVLogger]
        splitter = cls.get_splitter(arch)
        learn = learner.Learner(
            databunch,
            model,
            opt_func=config["opt_func"],
            loss_func=config["loss_func"](),
            cbs=model_cb,
            splitter=splitter,
        )
        learn.create_opt()
        learn.freeze()
        return learn

    @classmethod
    def get_augmentation_fn(cls, aug_name):
        if aug_name == "no_aug":
            return transforms.ColReader
        elif aug_name == "rules":
            augmenter = augmentations.RuleBasedAugmenter()
        elif aug_name == "LM":
            augmenter = augmentations.MLMSubstitutionAugmenter()
        else:
            raise ValueError(f"{aug_name} is not a supported augmentation mode")
        return partial(augmentations.AugmenterWrapper, augmenter=augmenter)

    def get_dataset(self, ds_type):
        if 'train' == ds_type:
            data = self.learn.dls.train
        elif 'valid' == ds_type:
            data = self.learn.dls.valid
        else:
            raise ValueError()
        return data

    def get_preds_dataset(self):
        return pd.DataFrame(columns=["ds_type", "text", "pred", "target"])

    def get_predictions(self, ds_type, tokenizer):
        data = self.get_dataset(ds_type)
        comb_df = self.get_preds_dataset()
        for sample in data:
            with torch.no_grad():
                new_df = self.get_preds_dataset()
                preds = self.learn.model.forward(sample[0])
                txt, preds, targets = self.get_batch_predictions(preds, sample, tokenizer)
                for i, batch_pred in enumerate(zip(txt, preds, targets)):
                    new_df.loc[i] = [ds_type] + list(batch_pred)
            comb_df = pd.concat([comb_df, new_df])
        comb_df = comb_df.reset_index(drop=True)
        return comb_df


class QuestionAnsweringPipeline(BlurrPipeline):
    def get_batch_predictions(self, preds, sample, tokenizer):
        x, target_start, target_end = sample
        preds_start, preds_end = preds.start_logits.argmax(dim=1).tolist(), preds.end_logits.argmax(
            dim=1).tolist()
        preds = [(s, e) for s, e in zip(preds_start, preds_end)]
        target_start, target_end = target_start.tolist(), target_end.tolist()
        targets = [(s, e) for s, e in zip(target_start, target_end)]
        txt = [tokenizer.decode(tokens, skip_special_tokens=True) for tokens in x['input_ids']]
        return txt, preds, targets

    @classmethod
    def get_splitter(cls, arch, **kwargs):
        return model_core.hf_splitter

    @classmethod
    def get_callbacks(cls, pre_config, config):
        return [question_anwsering.HF_QstAndAnsModelCallbackWithMetrics]

    @classmethod
    def get_databunch_from_name(cls, ds, aug_fn, arch, tokenizer, params):
        vocab = list(range(params["max_len"]))
        df = cls.load_data(ds, params["train_samples"])
        processed_df = data_processing.processing_from_name(df, ds, tokenizer, params["max_len"])

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

        get_question_auged = aug_fn(params["x_col"][0])
        get_context_auged = aug_fn(params["x_col"][1])


        get_x = partial(
            training_utils.get_qa_x,
            aug_question_fn=get_question_auged,
            aug_context_fn=get_context_auged,
            tokenizer=tokenizer
        )

        dblock = block.DataBlock(blocks=blocks,
                           get_x=get_x,
                           get_y=[transforms.ColReader(params["y_col"][0]), transforms.ColReader(params["y_col"][1])],
                           splitter=transforms.ColSplitter(col="is_valid"),
                           num_workers=0,
                           n_inp=1)

        dls = dblock.dataloaders(processed_df, bs=params["bs"])
        return dls


class SummarizationPipeline(BlurrPipeline):
    def get_batch_predictions(self, preds, sample, tokenizer):
        x, target = sample
        preds = preds.logits.argmax(dim=-1)
        txt = [tokenizer.decode(tokens, skip_special_tokens=True) for tokens in x['input_ids']]
        target_txt = [tokenizer.decode(tokens, skip_special_tokens=True) for tokens in target]
        pred_txt = [tokenizer.decode(tokens, skip_special_tokens=True) for tokens in preds]
        return txt, pred_txt, target_txt

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
        df = cls.load_data(ds, params["train_samples"])
        processed_df = data_processing.processing_from_name(df, ds, tokenizer, params["max_len"])

        hf_batch_tfm = model_sum.HF_SummarizationBatchTransform(
            arch,
            tokenizer,
            max_length=params["max_len"]
        )
        blocks = (data_core.HF_TextBlock(hf_batch_tfm=hf_batch_tfm), imports.noop)
        get_x = aug_fn(params["x_col"])

        dblock = block.DataBlock(blocks=blocks,
                           get_x=get_x,
                           get_y=transforms.ColReader(params["y_col"]),
                           splitter=transforms.ColSplitter(col="is_valid"))

        dls = dblock.dataloaders(processed_df, bs=params["bs"])
        return dls
