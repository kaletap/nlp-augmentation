from pathlib import Path


def get_lr(lr, div1, div2):
    return slice(lr / div1, lr / div2)


def get_qa_x(x, aug_question_fn, aug_context_fn, tokenizer):
    return (aug_question_fn(x), aug_context_fn(x)) \
        if (tokenizer.padding_side == 'right') \
        else (aug_context_fn(x), aug_question_fn(x))


def fill_paths(task, train_samples, aug, seed, config,
               paths=("model_save_paths", "metrics_save_paths", "predictions_save_paths", "targets_save_paths")):
    for path in paths:
        config[path] = Path(config[path].format(
            task=task,
            dataset="_".join(config["ds_name"]),
            model=config["pretrained_model_name"].replace("/", "-"),
            train_samples=str(train_samples),
            aug=aug,
            repeat=config["repeat"],
            seed=str(seed)),
        )
    return config
