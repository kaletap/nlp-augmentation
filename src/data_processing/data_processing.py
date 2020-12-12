import ast
from functools import partial


def processing_from_name(df, ds_name, tokenizer, max_len):
    if "squad" in ds_name:
        df = squad_v2_preprocessing(df, tokenizer, max_len)
    return df


def squad_v2_preprocessing(df, tokenizer, max_len):
    df['answers'] = df['answers'].map(ast.literal_eval)

    print(f"before preprocessing df.shape {df.shape}, {df['context'].str.split().str.len().max()}")
    params = tokenizer, max_len
    df = df.apply(partial(fixed_pre_process_squad, params=params), axis=1)
    print(f"after preprocessing df.shape {df.shape}, {df['context'].str.split().str.len().max()}")
    return df


def fixed_pre_process_squad(row, params):
    hf_tokenizer, max_len = params
    context, qst, ans = row['context'], row['question'], row['answers']

    tok_kwargs = {}
    if (hasattr(hf_tokenizer, 'add_prefix_space')): tok_kwargs['add_prefix_space'] = True

    if(hf_tokenizer.padding_side == 'right'):
        tok_input = hf_tokenizer.convert_ids_to_tokens(hf_tokenizer.encode(qst, context, **tok_kwargs))
    else:
        tok_input = hf_tokenizer.convert_ids_to_tokens(hf_tokenizer.encode(context, qst, **tok_kwargs))

    seq_len = len(tok_input)
    if seq_len > max_len:
        trim = (seq_len - max_len) // 2
        tok_input = tok_input[trim:]
        tok_input = tok_input[:-trim]

    start_idx, end_idx = 0, 0
    if not (ans['answer_start'] == 0 and len(ans['text']) == 0):
        tok_ans = hf_tokenizer.tokenize(str(row['answers']["text"][0]), **tok_kwargs)
        for idx, tok in enumerate(tok_input):
            try:
                if (tok == tok_ans[0] and tok_input[idx:idx + len(tok_ans)] == tok_ans):
                    start_idx, end_idx = idx, idx + len(tok_ans)
                    break
            except: pass

    print(f"len(tok_input) {len(tok_input)}")

    row['tokenized_input'] = tok_input
    row['tokenized_input_len'] = len(tok_input)
    row['tok_answer_start'] = start_idx
    row['tok_answer_end'] = end_idx

    return row
