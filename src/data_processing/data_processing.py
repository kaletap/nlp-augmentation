import ast
from functools import partial


def processing_from_name(df, ds_name, tokenizer, max_len):
    if ds_name[0] == "squad_v2":
        df = squad_v2_preprocessing(df, tokenizer, max_len)
    return df


def squad_v2_preprocessing(df, tokenizer, max_len):
    df['answers'] = df['answers'].map(ast.literal_eval)
    df["is_impossible"] = df["answers"].apply(lambda x: len(x["answer_start"]) == 0)
    df = df[df.is_impossible == False]

    df = df.apply(partial(fixed_pre_process_squad, hf_tokenizer=tokenizer), axis=1)
    df = df[df["tokenized_input"].apply(lambda x: len(x) < max_len)]
    return df


def fixed_pre_process_squad(row, hf_tokenizer):
    context, qst, ans = row['context'], row['question'], row['answers']

    tok_kwargs = {}
    if (hasattr(hf_tokenizer, 'add_prefix_space')): tok_kwargs['add_prefix_space'] = True

    if(hf_tokenizer.padding_side == 'right'):
        tok_input = hf_tokenizer.convert_ids_to_tokens(hf_tokenizer.encode(qst, context, **tok_kwargs))
    else:
        tok_input = hf_tokenizer.convert_ids_to_tokens(hf_tokenizer.encode(context, qst, **tok_kwargs))

    tok_ans = hf_tokenizer.tokenize(str(row['answers']["text"][0]), **tok_kwargs)

    start_idx, end_idx = 0, 0
    for idx, tok in enumerate(tok_input):
        try:
            if (tok == tok_ans[0] and tok_input[idx:idx + len(tok_ans)] == tok_ans):
                start_idx, end_idx = idx, idx + len(tok_ans)
                break
        except: pass

    row['tokenized_input'] = tok_input
    row['tokenized_input_len'] = len(tok_input)
    row['tok_answer_start'] = start_idx
    row['tok_answer_end'] = end_idx

    return row

