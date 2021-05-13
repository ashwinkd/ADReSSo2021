import re

import pandas as pd

dys_and_parse = pd.read_pickle('transcript_with_disfluency_parse.pickle')


def get_consecutive(tags):
    result = []
    i = 0
    while i < len(tags):
        if tags[i] != "E":
            i += 1
            continue
        error_word = (i, i + 1)
        i2 = i + 1
        while i2 < len(tags):
            if tags[i2] != "E":
                break
            error_word = (i, i2 + 1)
            i2 += 1
        result.append(error_word)
        i = i2

    return result


def normalize(error_text):
    error_text = error_text.lower()
    error_text = re.sub(r"\[.*\]", "", error_text)
    error_text = re.sub(r"[^a-z' ]", "", error_text)
    error_text = re.sub(r"\s+", " ", error_text).strip()
    return error_text


def make_unique(err_list):
    all_errors = []
    errors = {}
    for elem in err_list:
        if elem not in errors:
            errors[elem] = 0
        errors[elem] += 1
    for word, occ in errors.items():
        for i in range(occ):
            all_errors.append(f"{word}{i + 1}")
    return all_errors


def get_recall(all_errors, detected_errors):
    if all_errors is None or not all_errors:
        return 1
    if detected_errors is None:
        return 0
    found_errors = set(detected_errors).intersection(set(all_errors))
    recall = len(found_errors) / len(all_errors)
    return recall


def get_all_errors(target):
    if not re.findall(r"(\[\/\]|\[\/\/\]|\[x)", target):
        return None
    repeat_errors = []
    retrace_errors = []
    polysyllabic_errors = []
    for match in re.finditer(r"(\[\/\]|\[\/\/\]|\[x)", target):
        start = match.start()
        end = match.end()
        error_type = target[start: end]
        if start > 2:
            if target[start - 2] == ">":
                before_text = target[:start][::-1]
                error_s = before_text.index(">")
                error_e = before_text.index("<")
                error_text = before_text[error_s + 1:error_e][::-1]
            else:
                before_text = target[:start][::-1]
                error_s = before_text.index(" ")
                error_e = 0
                for space_m in re.finditer(r"\s", before_text[error_s + 1:]):
                    error_e = space_m.start()
                    break
                error_text = before_text[error_s + 1:error_e + 1][::-1]
            error_text = normalize(error_text)
            if error_type == "[/]":
                repeat_errors.append(error_text)
            elif error_type == '[//]':
                retrace_errors.append(error_text)
            elif error_type == '[x':
                polysyllabic_errors.append(error_text)
    all_errors = make_unique(repeat_errors + retrace_errors + polysyllabic_errors)
    return all_errors


dys_and_parse['all_errors'] = dys_and_parse.transcript_with_tags.apply(lambda x: get_all_errors(x))


def get_detected_errors(pred):
    if pred is None:
        return None
    pred = pred.split()
    pred_words = pred[::2]
    pred_tag = pred[1::2]
    detected_errors = []
    for start, end in get_consecutive(pred_tag):
        error_text = ""
        for w in pred_words[start:end]:
            if w.startswith("'"):
                error_text += w
            else:
                error_text += " " + w
        error_text = normalize(error_text)
        detected_errors.append(error_text)
    detected_errors = make_unique(detected_errors)
    return detected_errors


dys_and_parse['detected_errors'] = dys_and_parse.disfluency_text.apply(lambda x: get_detected_errors(x))

dys_and_parse['recall'] = dys_and_parse.apply(lambda x: get_recall(x.all_errors, x.detected_errors), axis=1)


def get_phase(x):
    if x in ["Control", "Dementia"]:
        return "TRAIN"
    else:
        return "TEST"


dys_and_parse['test_or_train'] = dys_and_parse.dx.apply(lambda x: get_phase(x))

dys_and_parse_test = dys_and_parse[dys_and_parse['test_or_train'] == "TEST"]
dys_and_parse_train = dys_and_parse[dys_and_parse['test_or_train'] == "TRAIN"]
dys_and_parse_train.to_pickle('transcript_with_disfluency_parse_train.pickle')
dys_and_parse_test.to_pickle('transcript_with_disfluency_parse_test.pickle')

recalls = dys_and_parse['recall'].tolist()
avg_recall = sum(recalls) / len(recalls)
print("Overall:", avg_recall)

recalls = dys_and_parse_train['recall'].tolist()
avg_recall = sum(recalls) / len(recalls)
print("Train: ", avg_recall)

recalls = dys_and_parse_test['recall'].tolist()
avg_recall = sum(recalls) / len(recalls)
print("Test:", avg_recall)
