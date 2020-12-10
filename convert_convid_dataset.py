import re
import os
import json
import pandas as pd

FOLDER = "./data/"
JSON_PATH = "./data/COVID-QA.json"
CSV_PATH = "COVID-QA.csv"
valid_size = 0.25

with open(JSON_PATH) as f:
    data = json.load(f)


def clean_text(text):
    text = text.replace("\n", " ")
    text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub(r'\[[^)]*\]', '', text)
    text = re.sub(r'http\S+', '', text)
    return text


start_texts = [
    "Text:",
    "The authors declare that they had full access to all of the data in this study and the\nauthors take complete responsibility for the integrity of the data and the accuracy of\nthe data analysis",
    "Jeffery K. Taubenberger\" and David M. Morens1-",
    "Updated March 21, 2020",
    "COVID-19 Update From China",
    "Abstract",
    "Summary",
]

end_texts = [
  "REFERENCES",
  "References",
  "Acknowledgments",
]


def trim_text(text, ans_start):
    fail = False
    for start in start_texts:
        text, ans_start, text_start = trim_intro(text, ans_start, start)
        if text_start >= 0:
            break
    else:
        fail = True

    for end in end_texts:
        text = trim_end(text, end)

    return text, ans_start, fail


def trim_intro(text, ans_start, start_text):
    text_start = text.find(start_text)
    if text_start >= 0:
        text_start += len(start_text)
        if ans_start > text_start:
            text = text[text_start:]
            ans_start -= text_start
    return text, ans_start, text_start


def trim_end(text, end_text):
    text_end = text.rfind(end_text)
    if text_end > 0:
        text = text[:text_end]
    return text


# data["data"][0]["paragraphs"][0].keys() # 'qas', 'context', 'document_id'
# data["data"][0]["paragraphs"][0]["qas"][0].keys() # 'question', 'id', 'answers'
# answers -> text, answer_start, is_impossible

# train test split on paragraphs

size = int(len(data["data"]) * valid_size)
train = data["data"][size:]
valid = data["data"][:size]

splits = [
    ("train_", train),
    ("validation_", valid),
]


for split, split_data in splits:
    correct_counter = 0
    fail_counter = 0
    csv_dict = {"context": [], "question": [], "answers": [], "is_impossible": []}
    for row in split_data:
        par = row["paragraphs"][0]
        for q in par["qas"]:
            text = par["context"]
            ans = q["answers"][0]["text"]
            ans_start = q["answers"][0]["answer_start"]

            text, ans_start, fail = trim_text(text, ans_start)
            ans_end = ans_start + len(ans)

            context_first_part = " ".join(text[:ans_start].strip().split())
            context_second_part = " ".join(text[ans_start + len(ans):].strip().split())

            clean_ans = clean_text(ans)
            clean_context_first_part = clean_text(context_first_part) + " "
            clean_context_second_part = " " + clean_text(context_second_part)
            clean_txt = clean_context_first_part + clean_ans + clean_context_second_part

            ans_start = len(clean_context_first_part)

            if fail:
                fail_counter += 1
            else:
                correct_counter += 1

            csv_dict["context"].append(clean_txt)
            csv_dict["question"].append(q["question"])
            csv_dict["answers"].append(str({"answer_start": [ans_start], "text": [clean_ans]}))
            csv_dict["is_impossible"].append(q["is_impossible"])

    print(f"correct_counter {correct_counter}")
    print(f"fail_counter {fail_counter}")

    path = os.path.join(FOLDER, split + CSV_PATH)
    pd.DataFrame(csv_dict).to_csv(path)
