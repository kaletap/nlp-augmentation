def get_lr(lr, div1, div2):
    return slice(lr / div1, lr / div2)


def get_qa_x(x, aug_question_fn, aug_context_fn, tokenizer):
    return (aug_question_fn(x), aug_context_fn(x)) \
        if (tokenizer.padding_side == 'right') \
        else (aug_context_fn(x), aug_question_fn(x))
