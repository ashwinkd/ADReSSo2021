import re

import pandas as pd

dys_and_parse = pd.read_pickle('transcript_with_disfluency_parse.pickle')


def get_combinations(error_words):
    if error_words[1] - error_words[0] < 2:
        return [error_words]
    error_words_list = list(range(error_words[0], error_words[1] + 1))
    length = len(error_words_list)
    list_of_combinations = [[error_words_list[i:] for i in range(l)] for l in range(2, length)]
    all_combinations = []
    for lcomb in list_of_combinations:
        lcomb = list(zip(*lcomb))
        for l in lcomb:
            all_combinations.append((min(l), max(l)))
    all_combinations.append(error_words)
    return all_combinations


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
        result += get_combinations(error_word)
        i = i2

    return result


def normalize(error_text):
    error_text = error_text.lower()
    error_text = re.sub(r"\[.*\]", "", error_text)
    error_text = re.sub(r"[^a-z' ]", "", error_text)
    error_text = re.sub(r"\s+", " ", error_text).strip()
    return error_text


def make_unique(word_repeat_errors, phrase_repeat_errors, word_retrace_errors, phrase_retrace_errors):
    phrase_repeat_errors += word_repeat_errors
    phrase_retrace_errors += word_retrace_errors
    error_type = ["WREP" for _ in word_repeat_errors] + \
                 ["PREP" for _ in phrase_repeat_errors] + \
                 ["WRET" for _ in word_retrace_errors] + \
                 ["PRET" for _ in phrase_retrace_errors]
    err_list = word_repeat_errors + phrase_repeat_errors + word_retrace_errors + phrase_retrace_errors
    all_errors = []
    for err_type, err_text in zip(error_type, err_list):
        num = 1
        elem = f"{err_text}|{err_type}|{num}"
        while elem in all_errors:
            num += 1
            elem = f"{err_text}|{err_type}|{num}"
        all_errors.append(elem)
    return all_errors


def get_accuracy(all_errors, detected_errors):
    if all_errors is None or not all_errors:
        return [-1, -1, -1, -1]
    if detected_errors is None:
        detected_errors = []
    target_WREP = []
    target_PREP = []
    target_WRET = []
    target_PRET = []
    pred_WREP = []
    pred_PREP = []
    pred_WRET = []
    pred_PRET = []
    for a in all_errors:
        if "WREP" in a:
            target_WREP.append(a)
        elif "PREP" in a:
            target_PREP.append(a)
        elif "WRET" in a:
            target_WRET.append(a)
        elif "PRET" in a:
            target_PRET.append(a)
    for d in detected_errors:
        if "WREP" in d:
            pred_WREP.append(d)
        elif "PREP" in d:
            pred_PREP.append(d)
        elif "WRET" in d:
            pred_WRET.append(d)
        elif "PRET" in d:
            pred_PRET.append(d)
    if len(target_WREP) > 0:
        acc_WREP = len(set(pred_WREP).intersection(set(target_WREP))) / len(target_WREP)
    else:
        acc_WREP = -1.0
    if len(target_PREP) > 0:
        acc_PREP = len(set(pred_PREP).intersection(set(target_PREP))) / len(target_PREP)
    else:
        acc_PREP = -1.0
    if len(target_WRET) > 0:
        acc_WRET = len(set(pred_WRET).intersection(set(target_WRET))) / len(target_WRET)
    else:
        acc_WRET = -1.0
    if len(target_PRET) > 0:
        acc_PRET = len(set(pred_PRET).intersection(set(target_PRET))) / len(target_PRET)
    else:
        acc_PRET = -1.0
    return [acc_WREP, acc_PREP, acc_WRET, acc_PRET]


def get_all_errors(target):
    if not re.findall(r"(\[\/\]|\[\/\/\]|\[x)", target):
        return None
    word_repeat_errors = []
    phrase_repeat_errors = []
    word_retrace_errors = []
    phrase_retrace_errors = []
    for match in re.finditer(r"(\[\/\]|\[\/\/\])", target):
        start = match.start()
        end = match.end()
        error_type = target[start: end]
        if start > 2:
            if target[start - 2] == ">":
                before_text = target[:start][::-1]
                error_s = before_text.index(">")
                error_e = before_text.index("<")
                error_text = before_text[error_s + 1:error_e][::-1]
                error_text = normalize(error_text)
                if error_type == "[/]":
                    phrase_repeat_errors.append(error_text)
                elif error_type == '[//]':
                    phrase_retrace_errors.append(error_text)
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
                    word_repeat_errors.append(error_text)
                elif error_type == '[//]':
                    word_retrace_errors.append(error_text)
    for match in re.finditer(r"\[x", target):
        start = match.start()
        end = match.end()
        before_text = target[:start][::-1]
        error_s = before_text.index(" ")
        error_e = 0
        for space_m in re.finditer(r"\s", before_text[error_s + 1:]):
            error_e = space_m.start()
            break
        error_text = before_text[error_s + 1:error_e + 1][::-1]
        error_text = normalize(error_text)
        for num in re.findall(r"\d+", target[end:end + 5]):
            word_repeat_errors += [error_text for _ in range(int(num))]
            break
    all_errors = make_unique(word_repeat_errors, phrase_repeat_errors, word_retrace_errors, phrase_retrace_errors)
    if not all_errors:
        return None
    return all_errors


dys_and_parse['all_errors'] = dys_and_parse.transcript_with_tags.apply(lambda x: get_all_errors(x))


def get_detected_errors(pred):
    if pred is None:
        return None
    pred = pred.split()
    pred_words = pred[::2]
    pred_tag = pred[1::2]
    detected_phrase_errors = []
    detected_word_errors = []
    for start, end in get_consecutive(pred_tag):
        error_text = ""
        for w in pred_words[start:end]:
            if w.startswith("'"):
                error_text += w
            else:
                error_text += " " + w
        error_text = normalize(error_text)
        detected_phrase_errors.append(error_text)
    detected_errors = make_unique(detected_phrase_errors.copy(),
                                  detected_word_errors.copy(),
                                  detected_phrase_errors.copy(),
                                  detected_word_errors.copy())
    return detected_errors


dys_and_parse['detected_errors'] = dys_and_parse.disfluency_text.apply(lambda x: get_detected_errors(x))

dys_and_parse['accuracy'] = dys_and_parse.apply(lambda x: get_accuracy(x.all_errors, x.detected_errors), axis=1)
dys_and_parse[['acc_WREP', 'acc_PREP', 'acc_WRET', 'acc_PRET']] = pd.DataFrame(dys_and_parse.accuracy.tolist(),
                                                                               index=dys_and_parse.index)


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

recalls = [r for r in dys_and_parse['acc_WREP'].tolist() if r != -1]
avg_acc = sum(recalls) / len(recalls)
print("Overall word repetition:", avg_acc)
recalls = [r for r in dys_and_parse['acc_PREP'].tolist() if r != -1]
avg_acc = sum(recalls) / len(recalls)
print("Overall phrase repetition:", avg_acc)
recalls = [r for r in dys_and_parse['acc_WRET'].tolist() if r != -1]
avg_acc = sum(recalls) / len(recalls)
print("Overall word retracing:", avg_acc)
recalls = [r for r in dys_and_parse['acc_PRET'].tolist() if r != -1]
avg_acc = sum(recalls) / len(recalls)
print("Overall phrase retracing:", avg_acc)

recalls = [r for r in dys_and_parse_train['acc_WREP'].tolist() if r != -1]
avg_acc = sum(recalls) / len(recalls)
print("Train word repetition:", avg_acc)
recalls = [r for r in dys_and_parse_train['acc_PREP'].tolist() if r != -1]
avg_acc = sum(recalls) / len(recalls)
print("Train phrase repetition:", avg_acc)
recalls = [r for r in dys_and_parse_train['acc_WRET'].tolist() if r != -1]
avg_acc = sum(recalls) / len(recalls)
print("Train word retracing:", avg_acc)
recalls = [r for r in dys_and_parse_train['acc_PRET'].tolist() if r != -1]
avg_acc = sum(recalls) / len(recalls)
print("Train phrase retracing:", avg_acc)

recalls = [r for r in dys_and_parse_test['acc_WREP'].tolist() if r != -1]
avg_acc = sum(recalls) / len(recalls)
print("Test word repetition:", avg_acc)
recalls = [r for r in dys_and_parse_test['acc_PREP'].tolist() if r != -1]
avg_acc = sum(recalls) / len(recalls)
print("Test phrase repetition:", avg_acc)
recalls = [r for r in dys_and_parse_test['acc_WRET'].tolist() if r != -1]
avg_acc = sum(recalls) / len(recalls)
print("Test word retracing:", avg_acc)
recalls = [r for r in dys_and_parse_test['acc_PRET'].tolist() if r != -1]
avg_acc = sum(recalls) / len(recalls)
print("Test phrase retracing:", avg_acc)

# def get_repeat_errors(target):
#     if not re.findall(r"\[\/\]", target):
#         return None
#     word_repeat_errors = []
#     phrase_repeat_errors = []
#     for match in re.finditer(r"\[\/\]", target):
#         start = match.start()
#         if start > 2:
#             if target[start - 2] == ">":
#                 before_text = target[:start][::-1]
#                 error_s = before_text.index(">")
#                 error_e = before_text.index("<")
#                 error_text = before_text[error_s + 1:error_e][::-1]
#                 error_text = normalize(error_text)
#                 phrase_repeat_errors.append(error_text)
#             else:
#                 before_text = target[:start][::-1]
#                 error_s = before_text.index(" ")
#                 error_e = 0
#                 for space_m in re.finditer(r"\s", before_text[error_s + 1:]):
#                     error_e = space_m.start()
#                     break
#                 error_text = before_text[error_s + 1:error_e + 1][::-1]
#                 error_text = normalize(error_text)
#                 word_repeat_errors.append(error_text)
#     for match in re.finditer(r"\[x", target):
#         start = match.start()
#         end = match.end()
#         before_text = target[:start][::-1]
#         error_s = before_text.index(" ")
#         error_e = 0
#         for space_m in re.finditer(r"\s", before_text[error_s + 1:]):
#             error_e = space_m.start()
#             break
#         error_text = before_text[error_s + 1:error_e + 1][::-1]
#         error_text = normalize(error_text)
#         for num in re.findall(r"\d+", target[end:end + 5]):
#             word_repeat_errors += [error_text for _ in range(int(num))]
#             break
#     word_repeat_errors = make_unique(word_repeat_errors)
#     phrase_repeat_errors = make_unique(phrase_repeat_errors)
#     return word_repeat_errors, phrase_repeat_errors
#
#
# def get_retrace_errors(target):
#     if not re.findall(r"\[\/\/\]", target):
#         return None
#     word_retrace_errors = []
#     phrase_retrace_errors = []
#     for match in re.finditer(r"\[\/\/\]", target):
#         start = match.start()
#         if start > 2:
#             if target[start - 2] == ">":
#                 before_text = target[:start][::-1]
#                 error_s = before_text.index(">")
#                 error_e = before_text.index("<")
#                 error_text = before_text[error_s + 1:error_e][::-1]
#                 error_text = normalize(error_text)
#                 phrase_retrace_errors.append(error_text)
#             else:
#                 before_text = target[:start][::-1]
#                 error_s = before_text.index(" ")
#                 error_e = 0
#                 for space_m in re.finditer(r"\s", before_text[error_s + 1:]):
#                     error_e = space_m.start()
#                     break
#                 error_text = before_text[error_s + 1:error_e + 1][::-1]
#                 error_text = normalize(error_text)
#                 word_retrace_errors.append(error_text)
#     word_retrace_errors = make_unique(word_retrace_errors)
#     phrase_retrace_errors = make_unique(phrase_retrace_errors)
#     return word_retrace_errors, phrase_retrace_errors
