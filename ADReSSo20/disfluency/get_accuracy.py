import re

import pandas as pd


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


def get_all_errors(target):
    if '[//]' not in target and '[/]' not in target:
        return None
    word_repeat_errors = []
    phrase_repeat_errors = []
    word_retrace_errors = []
    phrase_retrace_errors = []
    repeats = [index for index, element in enumerate(target) if element == '[/]']
    retraces = [index for index, element in enumerate(target) if element == '[//]']
    for rep_i in repeats:
        if rep_i < 2:
            continue
        if ">" in target[rep_i - 1]:
            error_i = [rep_i - 1]
            idx = rep_i - 1
            while idx > 0:
                t = target[idx]
                if idx not in error_i:
                    error_i = [idx] + error_i
                if "<" in t:
                    break
                idx -= 1
            phrase_repeat_errors.append(error_i)
        else:
            error_i = [rep_i - 1]
            word_repeat_errors.append(error_i)
            phrase_repeat_errors.append(error_i)
    for ret_i in retraces:
        if ret_i < 2:
            continue
        if ">" in target[ret_i - 1]:
            error_i = [ret_i - 1]
            idx = ret_i - 1
            while idx > 0:
                t = target[idx]
                if idx not in error_i:
                    error_i = [idx] + error_i
                if "<" in t:
                    break
                idx -= 1
            phrase_retrace_errors.append(error_i)
        else:
            error_i = [ret_i - 1]
            word_retrace_errors.append(error_i)
            phrase_retrace_errors.append(error_i)
    if not word_repeat_errors and not phrase_repeat_errors and not word_retrace_errors and not phrase_retrace_errors:
        return None
    return word_repeat_errors, phrase_repeat_errors, word_retrace_errors, phrase_retrace_errors


def get_tokens(text):
    if text is None:
        return []
    text = text.split()
    twords = text[::2]
    return twords


def resolve_text(transcript_with_tags):
    transcript_with_tags = transcript_with_tags.lower()
    transcript_with_tags = transcript_with_tags.replace("'s", " 's")
    transcript_with_tags = transcript_with_tags.replace("'m", " 'm")
    transcript_with_tags = transcript_with_tags.replace("'re", " 're")
    transcript_with_tags = transcript_with_tags.replace("'ll", " 'll")
    transcript_with_tags = transcript_with_tags.replace("'ve", " 've")
    transcript_with_tags = transcript_with_tags.replace("'d", " 'd")
    transcript_with_tags = transcript_with_tags.replace("_", " ")
    transcript_with_tags = transcript_with_tags.replace(":", " ")
    transcript_with_tags = transcript_with_tags.replace("+", " ")
    transcript_with_tags = transcript_with_tags.replace("n't", " n't")
    if re.findall(r"\[x[0-9 ]+\]", transcript_with_tags):
        transcript_with_tags = resolve_repeats(transcript_with_tags)
    transcript_with_tags = re.split(r"\s", transcript_with_tags)
    return transcript_with_tags


def resolve_repeats(text):
    text_tokens = text.split()
    idx = 0
    while idx < len(text_tokens):
        token = text_tokens[idx]
        if token != "[x":
            idx += 1
            continue
        repeat_count = int(re.sub(r"[^0-9]", "", text_tokens[idx + 1])) - 1
        repeats = []
        for _ in range(repeat_count):
            repeats += [text_tokens[idx - 1], "[/]"]
        if idx + 2 < len(text_tokens):
            text_tokens = text_tokens[:idx] + repeats + text_tokens[idx + 2:]
        else:
            text_tokens = text_tokens[:idx] + repeats
        idx += 1
    return " ".join(text_tokens)


def get_alignment(tokens, original_tokens):
    original_clean_tokens = [normalize(t) for t in original_tokens]
    starter = -1
    matched_indices = []
    for token in tokens:
        token = normalize(token)
        sub_original_tokens = [None for _ in range(starter + 1)] + original_clean_tokens[starter + 1:]
        token_index = sub_original_tokens.index(token)
        matched_indices.append(token_index)
        starter = token_index
    alignments = list(zip(tokens, matched_indices))
    return alignments


def get_acc(target, pred):
    if target is None or not target:
        return -1.0
    return len(pred) / len(target)


def get_accuracy(pred, alignment, all_errors):
    if all_errors is None or not all_errors:
        return [-1, -1, -1, -1]
    if pred is None or not pred:
        return [0, 0, 0, 0]
    pred = pred.split()
    detected_wrep_errors = {}
    detected_prep_errors = {}
    detected_wret_errors = {}
    detected_pret_errors = {}
    target_wreps, target_preps, target_wrets, target_prets = all_errors

    for align, tag in zip(alignment, pred[1::2]):
        word, orig_idx = align
        if tag != "E":
            continue
        for wrep in target_wreps:
            if orig_idx in wrep:
                key = "_".join([str(wi) for wi in wrep])
                val = [True if wi == orig_idx else False for wi in wrep]
                if key not in detected_wrep_errors:
                    detected_wrep_errors[key] = val
                else:
                    value = detected_wrep_errors[key]
                    value = [new or old for new, old in zip(val, value)]
                    detected_wrep_errors[key] = value
        for prep in target_preps:
            if orig_idx in prep:
                key = "_".join([str(wi) for wi in prep])
                val = [True if wi == orig_idx else False for wi in prep]
                if key not in detected_prep_errors:
                    detected_prep_errors[key] = val
                else:
                    value = detected_prep_errors[key]
                    value = [new or old for new, old in zip(val, value)]
                    detected_prep_errors[key] = value
        for wret in target_wrets:
            if orig_idx in wret:
                key = "_".join([str(wi) for wi in wret])
                val = [True if wi == orig_idx else False for wi in wret]
                if key not in detected_wret_errors:
                    detected_wret_errors[key] = val
                else:
                    value = detected_wret_errors[key]
                    value = [new or old for new, old in zip(val, value)]
                    detected_wret_errors[key] = value
        for pret in target_prets:
            if orig_idx in pret:
                key = "_".join([str(wi) for wi in pret])
                val = [True if wi == orig_idx else False for wi in pret]
                if key not in detected_pret_errors:
                    detected_pret_errors[key] = val
                else:
                    value = detected_pret_errors[key]
                    value = [new or old for new, old in zip(val, value)]
                    detected_pret_errors[key] = value
    success_wrep = []
    success_prep = []
    success_wret = []
    success_pret = []
    for key, value in detected_wrep_errors.items():
        if sum(value) == len(value):
            success_wrep.append(key)
    for key, value in detected_prep_errors.items():
        if sum(value) == len(value):
            success_prep.append(key)
    for key, value in detected_wret_errors.items():
        if sum(value) == len(value):
            success_wret.append(key)
    for key, value in detected_pret_errors.items():
        if sum(value) == len(value):
            success_pret.append(key)
    wrep_acc = get_acc(target_wreps, success_wrep)
    prep_acc = get_acc(target_preps, success_prep)
    wret_acc = get_acc(target_wrets, success_wret)
    pret_acc = get_acc(target_prets, success_pret)
    return [wrep_acc, prep_acc, wret_acc, pret_acc]


def get_phase(x):
    if x in ["Control", "Dementia"]:
        return "TRAIN"
    else:
        return "TEST"


def main():
    dys_and_parse = pd.read_pickle('transcript_with_disfluency_parse.pickle')
    dys_and_parse["tokens"] = dys_and_parse.disfluency_text.apply(lambda x: get_tokens(x))
    dys_and_parse['transcript_with_tags_tokens'] = dys_and_parse.transcript_with_tags.apply(lambda x: resolve_text(x))
    dys_and_parse["alignments"] = dys_and_parse.apply(lambda x: get_alignment(x.tokens, x.transcript_with_tags_tokens),
                                                      axis=1)

    dys_and_parse['all_errors'] = dys_and_parse.transcript_with_tags_tokens.apply(lambda x: get_all_errors(x))
    dys_and_parse['accuracy'] = dys_and_parse.apply(lambda x: get_accuracy(x.disfluency_text,
                                                                           x.alignments,
                                                                           x.all_errors), axis=1)
    dys_and_parse[['acc_WREP', 'acc_PREP', 'acc_WRET', 'acc_PRET']] = pd.DataFrame(dys_and_parse.accuracy.tolist(),
                                                                                   index=dys_and_parse.index)
    dys_and_parse['test_or_train'] = dys_and_parse.dx.apply(lambda x: get_phase(x))

    dys_and_parse_test = dys_and_parse[dys_and_parse['test_or_train'] == "TEST"]
    dys_and_parse_train = dys_and_parse[dys_and_parse['test_or_train'] == "TRAIN"]
    dys_and_parse.to_pickle('transcript_with_disfluency_parse.pickle')
    dys_and_parse_train.to_pickle('transcript_with_disfluency_parse_train.pickle')
    dys_and_parse_test.to_pickle('transcript_with_disfluency_parse_test.pickle')

    avg_accuracy = [r for r in dys_and_parse['acc_WREP'].tolist() if r != -1]
    avg_acc = sum(avg_accuracy) / len(avg_accuracy)
    print("Overall word repetition:", avg_acc)
    avg_accuracy = [r for r in dys_and_parse['acc_PREP'].tolist() if r != -1]
    avg_acc = sum(avg_accuracy) / len(avg_accuracy)
    print("Overall phrase repetition:", avg_acc)
    avg_accuracy = [r for r in dys_and_parse['acc_WRET'].tolist() if r != -1]
    avg_acc = sum(avg_accuracy) / len(avg_accuracy)
    print("Overall word retracing:", avg_acc)
    avg_accuracy = [r for r in dys_and_parse['acc_PRET'].tolist() if r != -1]
    avg_acc = sum(avg_accuracy) / len(avg_accuracy)
    print("Overall phrase retracing:", avg_acc)

    avg_accuracy = [r for r in dys_and_parse_train['acc_WREP'].tolist() if r != -1]
    avg_acc = sum(avg_accuracy) / len(avg_accuracy)
    print("Train word repetition:", avg_acc)
    avg_accuracy = [r for r in dys_and_parse_train['acc_PREP'].tolist() if r != -1]
    avg_acc = sum(avg_accuracy) / len(avg_accuracy)
    print("Train phrase repetition:", avg_acc)
    avg_accuracy = [r for r in dys_and_parse_train['acc_WRET'].tolist() if r != -1]
    avg_acc = sum(avg_accuracy) / len(avg_accuracy)
    print("Train word retracing:", avg_acc)
    avg_accuracy = [r for r in dys_and_parse_train['acc_PRET'].tolist() if r != -1]
    avg_acc = sum(avg_accuracy) / len(avg_accuracy)
    print("Train phrase retracing:", avg_acc)

    avg_accuracy = [r for r in dys_and_parse_test['acc_WREP'].tolist() if r != -1]
    avg_acc = sum(avg_accuracy) / len(avg_accuracy)
    print("Test word repetition:", avg_acc)
    avg_accuracy = [r for r in dys_and_parse_test['acc_PREP'].tolist() if r != -1]
    avg_acc = sum(avg_accuracy) / len(avg_accuracy)
    print("Test phrase repetition:", avg_acc)
    avg_accuracy = [r for r in dys_and_parse_test['acc_WRET'].tolist() if r != -1]
    avg_acc = sum(avg_accuracy) / len(avg_accuracy)
    print("Test word retracing:", avg_acc)
    avg_accuracy = [r for r in dys_and_parse_test['acc_PRET'].tolist() if r != -1]
    avg_acc = sum(avg_accuracy) / len(avg_accuracy)
    print("Test phrase retracing:", avg_acc)


if __name__ == '__main__':
    main()

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
# def get_accuracy(all_errors, detected_errors):
#     if all_errors is None or not all_errors:
#         return [-1, -1, -1, -1]
#     if detected_errors is None:
#         detected_errors = []
#     target_WREP = []
#     target_PREP = []
#     target_WRET = []
#     target_PRET = []
#     pred_WREP = []
#     pred_PREP = []
#     pred_WRET = []
#     pred_PRET = []
#     for a in all_errors:
#         if "WREP" in a:
#             target_WREP.append(a)
#         elif "PREP" in a:
#             target_PREP.append(a)
#         elif "WRET" in a:
#             target_WRET.append(a)
#         elif "PRET" in a:
#             target_PRET.append(a)
#     for d in detected_errors:
#         if "WREP" in d:
#             pred_WREP.append(d)
#         elif "PREP" in d:
#             pred_PREP.append(d)
#         elif "WRET" in d:
#             pred_WRET.append(d)
#         elif "PRET" in d:
#             pred_PRET.append(d)
#     if len(target_WREP) > 0:
#         acc_WREP = len(set(pred_WREP).intersection(set(target_WREP))) / len(target_WREP)
#     else:
#         acc_WREP = -1.0
#     if len(target_PREP) > 0:
#         acc_PREP = len(set(pred_PREP).intersection(set(target_PREP))) / len(target_PREP)
#     else:
#         acc_PREP = -1.0
#     if len(target_WRET) > 0:
#         acc_WRET = len(set(pred_WRET).intersection(set(target_WRET))) / len(target_WRET)
#     else:
#         acc_WRET = -1.0
#     if len(target_PRET) > 0:
#         acc_PRET = len(set(pred_PRET).intersection(set(target_PRET))) / len(target_PRET)
#     else:
#         acc_PRET = -1.0
#     return [acc_WREP, acc_PREP, acc_WRET, acc_PRET]
