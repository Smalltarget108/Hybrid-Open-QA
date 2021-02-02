import json
import jsonlines
import re
import collections
import string
import sys
from tqdm import tqdm
import pdb
from lib.query import Query

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


def t5_tokenization(text, activate=False):
    if activate:
        text = tokenizer.decode(tokenizer.encode(text))
    return text


def get_raw_scores(examples, reference=None):

    exact_scores = {}
    f1_scores = {}

    top_1_examples = []

    skip_step = 1

    for idx in range(0, len(examples), skip_step):

        curr_examples = examples[idx:idx+skip_step]

        # if 'answer' not in curr_examples[0]['gen_top1']:
        #     continue

        em = 0
        f1 = 0
        top_1 = curr_examples[0]
        for example in curr_examples:
            if ref:
                gold_answers = [t5_tokenization(str(x)) for x in reference[idx//skip_step]['denotation']]
                # qas_id = reference[idx//3]['qid']
                qas_id = idx//skip_step
            else:
                # gold_answers = [replace_keys(example['tgt'], 'answer')]
                gold_answers = [t5_tokenization(str(x)) for x in example['ans_text']]
                qas_id = idx//skip_step

            # prediction = example['exe_results'] if 'exe_results' in example else replace_keys(example['gen'], 'answer')
            prediction = example['exe_results'] if 'sql' in example['gen_top1'] else replace_keys(example['gen_top1'], 'answer')

            curr_em = em
            curr_f1 = f1

            if isinstance(prediction, str):
                prediction = t5_tokenization(prediction)
                em = max(em, max(compute_exact(a, prediction) for a in gold_answers))
                f1 = max(f1, max(compute_f1(a, prediction) for a in gold_answers))
            elif isinstance(prediction, list):
                if len(prediction) > 0:
                    em = max(em, max(compute_exact(a, t5_tokenization(str(pred))) for a in gold_answers for pred in prediction))
                    f1 = max(f1, max(compute_f1(a, t5_tokenization(str(pred))) for a in gold_answers for pred in prediction))
                else:
                    exact_scores[qas_id] = 0
                    f1_scores[qas_id] = 0
            elif isinstance(prediction, int) or isinstance(prediction, float):
                prediction = t5_tokenization(str(prediction))
                em = max(em, max(compute_exact(a, prediction) for a in gold_answers))
                f1 = max(f1, max(compute_f1(a, prediction) for a in gold_answers))
            else:
                raise ValueError(prediction)

            if em > curr_em:
                top_1 = example

        top_1_examples.append(top_1)

        exact_scores[qas_id] = em
        f1_scores[qas_id] = f1

    qid_list = exact_scores.keys()
    total = len(qid_list)

    with jsonlines.open(sys.argv[1] + '.top1', 'w') as writer:
        writer.write_all(top_1_examples)

    return collections.OrderedDict(
        [
            ("total exact", 100.0 * sum(exact_scores[k] for k in qid_list) / total),
            ("total f1", 100.0 * sum(f1_scores[k] for k in qid_list) / total),
            ("total", total),
        ]
    )


def lf_acc_for_one_line(prediction, reference):
    qp = Query.from_dict(prediction['durepa_full']['parsed_sql']['query'], ordered=False)
    qg = Query.from_dict(reference['parsed_sql']['query'], ordered=False)

    return qp == qg

def compute_lf_acc(predictions, references):
    lf_match = []
    for pred, ref in tqdm(zip(predictions, references), total=len(predictions)):
        if 'parsed_sql' not in pred['durepa_full'] or not pred['durepa_full']['success'] or not ref['success']:
            continue
        lf_match.append(lf_acc_for_one_line(pred, ref))

    return collections.OrderedDict(
        [
            ("total lf acc", 100.0 * sum(lf_match) / len(lf_match)),
            ("total", len(lf_match)),
        ]
    )


def sql_no_table_match(prediction, reference):
    pred_sql_str = prediction['durepa_full']['gen'].replace('sql:', '').lstrip()
    ref_sql_str = t5_tokenization(reference['true_sql'].lstrip(), True)
    pred_table = re.search(r'FROM(.*?)WHERE', pred_sql_str)
    ref_table = re.search(r'FROM(.*?)WHERE', ref_sql_str)
    if pred_table:
        pred_table = pred_table.group(1)
    else:
        pred_table = ''
    if ref_table:
        ref_table = ref_table.group(1)
    else:
        ref_table = ''

    pred_sql_str = pred_sql_str.replace(pred_table, '').lower()
    ref_sql_str = ref_sql_str.replace(ref_table, '').lower()

    if pred_sql_str != ref_sql_str:
        print(pred_sql_str)
        print(ref_sql_str)
        print()

    return pred_sql_str == ref_sql_str


def compute_sql_no_table_match(predictions, references):
    match = []
    for pred, ref in tqdm(zip(predictions, references), total=len(predictions)):
        if 'parsed_sql' not in pred['durepa_full'] or not pred['durepa_full']['success'] or not ref['success']:
            continue
        match.append(sql_no_table_match(pred, ref))

    return collections.OrderedDict(
        [
            ("total sql no table acc", 100.0 * sum(match) / len(match)),
            ("total", len(match)),
        ]
    )


assert len(sys.argv) >= 2, "you need to input the file"

with jsonlines.open(sys.argv[1], 'r') as f:
    data = [line for line in tqdm(f.iter())]

ref = None
# with jsonlines.open(sys.argv[2], 'r') as f:
#     ref = [line for line in tqdm(f.iter())]
#
# if len(sys.argv) == 4:
#     with jsonlines.open(sys.argv[3], 'r') as f:
#         ref.extend([line for line in tqdm(f.iter())])

print(get_raw_scores(data, ref))

# print(compute_lf_acc(data, ref))
#
# print(compute_sql_no_table_match(data, ref))
