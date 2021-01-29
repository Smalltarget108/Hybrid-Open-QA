import json
import jsonlines
import re
import collections
import string
import sys
from tqdm import tqdm
import pdb
from transformers import T5Tokenizer

tokenizer = T5Tokenizer.from_pretrained('t5-base')


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def replace_keys(text, key):
    key = key + ":"
    return text.replace(key, "").lstrip()


def remove_special_tokens(text):
    return re.sub('[^A-Za-z0-9]+', '', text)


def t5_tokenization(text):
    return tokenizer.decode(tokenizer.encode(text))


def get_raw_scores(examples):

    exact_scores = {}
    f1_scores = {}

    for idx, example in tqdm(enumerate(examples), total=len(examples)):

        if idx % 3 != 0:
            continue

        gold_answers = [t5_tokenization(replace_keys(x, 'answer')) for x in example['tgt']]
        qas_id = idx
        # if 'sql' in example['gen_top1']:
        #     prediction = example['exe_results']
        # elif 'answer' in example['gen_top1']:
        #     prediction = replace_keys(example['gen_top1'], "answer")
        # elif 'denotation' in example['gen_top1']:
        #     prediction = replace_keys(example['gen_top1'], "denotation")
        # else:
        #     raise NotImplementedError()

        prediction = example['exe_results'] if 'exe_results' in example else replace_keys(example['gen'], 'answer')

        if isinstance(prediction, str):
            prediction = t5_tokenization(prediction)
            exact_scores[qas_id] = max(compute_exact(a, prediction) for a in gold_answers)
            f1_scores[qas_id] = max(compute_f1(a, prediction) for a in gold_answers)
        elif isinstance(prediction, list):
            if len(prediction) > 0:
                exact_scores[qas_id] = max(compute_exact(a, t5_tokenization(str(pred))) for a in gold_answers for pred in prediction)
                f1_scores[qas_id] = max(compute_f1(a, t5_tokenization(str(pred))) for a in gold_answers for pred in prediction)
            else:
                exact_scores[qas_id] = 0
                f1_scores[qas_id] = 0
        elif isinstance(prediction, int) or isinstance(prediction, float):
            prediction = t5_tokenization(str(prediction))
            exact_scores[qas_id] = max(compute_exact(a, prediction) for a in gold_answers)
            f1_scores[qas_id] = max(compute_f1(a, prediction) for a in gold_answers)
        else:
            raise ValueError(prediction)

    qid_list = exact_scores.keys()
    total = len(qid_list)

    return collections.OrderedDict(
        [
            ("total exact", 100.0 * sum(exact_scores[k] for k in qid_list) / total),
            ("total f1", 100.0 * sum(f1_scores[k] for k in qid_list) / total),
            ("total", total),
        ]
    )


assert len(sys.argv) >= 2, "you need to input the file"

with jsonlines.open(sys.argv[1], 'r') as f:
    data = [line for line in tqdm(f.iter())]

# with jsonlines.open(sys.argv[2], 'r') as f:
#     ref = [line for line in tqdm(f.iter())]
#
# if len(sys.argv) == 4:
#     with jsonlines.open(sys.argv[3], 'r') as f:
#         ref.extend([line for line in tqdm(f.iter())])

print(get_raw_scores(data))
