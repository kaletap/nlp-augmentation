import argparse
from copy import deepcopy


from src.pipelines import configs
from src import training_utils


parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="qa",
                    choices=list(configs.experiments_setup["tasks"].keys()), help="type of task")
parser.add_argument("--data_path", type=str, default="",
                    choices=list(configs.experiments_setup["tasks"].keys()), help="path to folder with data")
args = parser.parse_args()


def run_exp(task, data_path, main_config):
    pipe_cls, og_config = main_config["tasks"][task]
    for seed in main_config["seeds"]:
        og_config["seed"] = seed
        for train_samples in main_config["train_samples"]:
            og_config["train_samples"] = train_samples
            for aug in main_config["augmentations"]:
                og_config["augmentation"] = aug
                config = deepcopy(og_config)
                config = training_utils.fill_paths(task, train_samples, aug, seed, config)
                pipe = pipe_cls.from_name(data_path=data_path, exp_parameters=config)
                pipe.run()


if __name__ == "__main__":
    run_exp(task=args.task, data_path=args.data_path, main_config=configs.experiments_setup)
