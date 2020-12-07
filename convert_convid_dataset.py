import os
import json
import pandas as pd

FOLDER = "./data/"
JSON_PATH = "./data/COVID-QA.json"
CSV_PATH = "COVID-QA.csv"
valid_size = 0.25
CONTEXT_MAX_LEN = 512

with open(JSON_PATH) as f:
    data = json.load(f)

# data["data"][0]["paragraphs"][0].keys() # 'qas', 'context', 'document_id'
# data["data"][0]["paragraphs"][0]["qas"][0].keys() # 'question', 'id', 'answers'
# answers -> text, answer_start, is_impossible

# train test split on paragraphs


# import pdb;pdb.set_trace()


size = int(len(data["data"]) * valid_size)
train = data["data"][size:]
valid = data["data"][:size]

splits = [
    ("train_", train),
    ("valid_", valid),
]

for split, split_data in splits:
    csv_dict = {"context": [], "question": [], "answers": [], "is_impossible": []}
    for row in split_data:
        par = row["paragraphs"][0]
        for q in par["qas"]:
            text = par["context"]
            ans = q["answers"][0]["text"]
            ans_start = q["answers"][0]["answer_start"]

            ans_word_count = len(ans.split())
            max_len = (CONTEXT_MAX_LEN - ans_word_count) // 2

            context_first_part = " ".join(text[:ans_start].split()[-max_len:])
            context_second_part = " ".join(text[ans_start + len(ans):].split()[:max_len])
            cropped_context = context_first_part + " " + ans + " " + context_second_part
            ans_start = len(context_first_part)

            csv_dict["context"].append(cropped_context)
            csv_dict["question"].append(q["question"])
            csv_dict["answers"].append(str({"answer_start": [ans_start], "text": [ans]}))
            csv_dict["is_impossible"].append(q["is_impossible"])

    path = os.path.join(FOLDER, split + CSV_PATH)
    pd.DataFrame(csv_dict).to_csv(path)
