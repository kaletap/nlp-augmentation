from copy import deepcopy
from pathlib import Path
from src.pipelines import configs


def fill_paths(task, train_samples, aug, seed, config):
    for path in ["model_save_paths", "metrics_save_paths", "predictions_save_paths", "targets_save_paths"]:
        config[path] = Path(config[path].format(
            task=task,
            dataset=config["ds_name"],
            model=config["pretrained_model_name"],
            train_samples=str(train_samples),
            aug=aug,
            seed=str(seed))
        )
    return config


def run_exp(task, main_config):
    pipe_cls, og_config = main_config["tasks"][task]
    for aug in main_config["augmentations"]:
        og_config["augmentation"] = aug
        for train_samples in main_config["train_samples"]:
            og_config["train_samples"] = train_samples
            for seed in main_config["seeds"]:
                og_config["seed"] = seed
                config = deepcopy(og_config)
                config = fill_paths(task, train_samples, aug, seed, config)
                pipe = pipe_cls.from_name(exp_parameters=config)
                pipe.run()


if __name__ == "__main__":
    run_exp(task="qa", main_config=configs.experiments_setup)
    # run_exp(task="summarization", main_config=configs.experiments_setup)
