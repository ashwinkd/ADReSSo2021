import re
from itertools import groupby

import pandas as pd

df = pd.read_csv('data.csv', usecols=['ID', 'mmse'])
id_to_mmse = df.set_index('ID')['mmse'].to_dict()

data = pd.read_pickle('../transcript_with_disfluency_parse_train.pickle')

data['mmse'] = data.speaker.apply(lambda x: id_to_mmse.get(x, None))


# classwise_Etags = {"Control": 0, "Dementia": 0}
# classwise_percentage = {"Control": 0, "Dementia": 0}
# classwise_count = {"Control": 0, "Dementia": 0}
# for (idx, (disfluency_text, dx)) in data[['disfluency_text', 'dx']].iterrows():
#     if disfluency_text is None:
#         continue
#     tokens = disfluency_text.split()
#     tags = tokens[1::2]
#     e_tokens = [t for t in tags if t == "E"]
#     classwise_Etags[dx] += len(e_tokens)
#     classwise_percentage[dx] += len(e_tokens) / len(tags)
#     classwise_count[dx] += 1
# print(classwise_Etags)
# print("Control AVG%: ", classwise_percentage["Control"]/classwise_count["Control"])
# print("Dementia AVG%: ", classwise_percentage["Dementia"]/classwise_count["Dementia"])
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z' ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def resolve_repeats(text):
    text_tokens = text.split()
    idx = 0
    while idx < len(text_tokens):
        token = text_tokens[idx]
        if token != "[x":
            idx += 1
            continue
        repeat_count = int(re.sub(r"[^0-9]", "", text_tokens[idx + 1])) - 1
        repeats = [text_tokens[idx - 1]] * repeat_count
        if idx + 2 < len(text_tokens):
            text_tokens = text_tokens[:idx] + repeats + text_tokens[idx + 2:]
        else:
            text_tokens = text_tokens[:idx] + repeats
        idx += 1
    return " ".join(text_tokens)


def remove_tags(text):
    text = text.replace("*PAR:\t", '')
    if text.endswith('...'):
        text = text[-3:]
    text = text.replace("*PAR:", '')
    text = text.replace("(...)", '')
    text = text.replace("...", '')
    text = text.replace("(..)", '')
    text = text.replace("(.)", '')
    text = text.replace('[//]', '')
    text = text.replace('[/]', '')
    text = text.replace('‡', '')
    text = text.replace('xxx', '')
    text = text.replace('[=! sings]', '')
    text = re.sub(r"[&()<>]", '', text)
    matches = re.findall(r'\[[\:\*][a-zA-Z\:\_\'\-\@\s]+\]', text)
    for match in matches:
        text = text.replace(match, '')
    if re.findall(r"\d+\_\d+", text):
        text = re.split(r'[\s/+"][.?!][\s[]', text)[0]
    text = re.sub(r"\d+\_\d+]", "", text)
    text = re.sub(r"\+", "", text)
    text = re.sub(r"\"", "", text)
    if text.startswith('.'):
        text = text[1:]
    text = re.sub(r"[=:][a-z]+", "", text)
    if re.findall(r"\[x[0-9 ]+\]", text):
        text = resolve_repeats(text)
    text = text.replace('[]', '')
    text = text.replace('_', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    text_tokens = text.split()
    text_tokens = [t if "@" not in t else "" for t in text_tokens]
    text = " ".join(text_tokens)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# data['transcript_without_tags'] = data.transcript_with_tags.apply(lambda x: remove_tags(x))
# data['transcript_without_tags'] = data.transcript_without_tags.apply(lambda x: clean_text(x))
def get_words(text):
    if text is None:
        return
    tokens = text.split()
    twords = tokens[::2]
    transcript = " ".join(twords)
    return transcript


data['transcript_without_tags'] = data.disfluency_text.apply(lambda x: get_words(x))


def remove_underscore(text):
    if text is None:
        return
    text = re.sub(r"\_", "F", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


data['transcript_without_underscore'] = data.disfluency_text.apply(lambda x: remove_underscore(x))


def update(new, old):
    t = None
    if new is not None:
        t = new
    if old is not None:
        t = old
    return t


def remove_repetition(pred, alignment, all_errors):
    if all_errors is None or not all_errors:
        if pred is not None:
            pred = pred.split()
            transcript = " ".join([w for w in pred[::2]])
            transcript = ""
            return transcript
        else:
            return None
    if pred is None or not pred:
        return None
    pred = pred.split()
    pred_tags = pred[1::2]
    pred_words = pred[::2]
    detected_wrep_errors = {}
    detected_wrep_idx = {}
    detected_prep_errors = {}
    detected_prep_idx = {}
    target_wreps, target_preps, _, _ = all_errors
    for (idx, (word, orig_idx), tag) in zip(list(range(len(alignment))), alignment, pred_tags):
        if tag != "E":
            continue
        for wrep in target_wreps:
            if orig_idx in wrep:
                key = "_".join([str(wi) for wi in wrep])
                val = [True if wi == orig_idx else False for wi in wrep]
                val2 = [idx if wi == orig_idx else None for wi in wrep]
                if key not in detected_wrep_errors:
                    detected_wrep_errors[key] = val
                    detected_wrep_idx[key] = val2
                else:
                    value = detected_wrep_errors[key]
                    value2 = detected_wrep_idx[key]
                    value = [new or old for new, old in zip(val, value)]
                    value2 = [update(new, old) for new, old in zip(val2, value2)]
                    detected_wrep_errors[key] = value
                    detected_wrep_idx[key] = value2
        for prep in target_preps:
            if orig_idx in prep:
                key = "_".join([str(wi) for wi in prep])
                val = [True if wi == orig_idx else False for wi in prep]
                val2 = [idx if wi == orig_idx else None for wi in prep]
                if key not in detected_prep_errors:
                    detected_prep_errors[key] = val
                    detected_prep_idx[key] = val2
                else:
                    value = detected_prep_errors[key]
                    value2 = detected_prep_idx[key]
                    value = [new or old for new, old in zip(val, value)]
                    value2 = [update(new, old) for new, old in zip(val2, value2)]
                    detected_prep_errors[key] = value
                    detected_prep_idx[key] = value2
    success_wrep = []
    success_prep = []
    for key, value in detected_wrep_errors.items():
        if sum(value) == len(value):
            success_wrep.append(key)
    for key, value in detected_prep_errors.items():
        if sum(value) == len(value):
            success_prep.append(key)
    for key in success_wrep:
        words_to_remove = detected_wrep_idx[key]
        for wi in words_to_remove:
            pred_words[wi] = "[TO BE REMOVED]"
    for key in success_prep:
        words_to_remove = detected_prep_idx[key]
        for wi in words_to_remove:
            pred_words[wi] = "[TO BE REMOVED]"
    transcript = " ".join([w for w in pred_words if w != '[TO BE REMOVED]'])
    return transcript


data['transcript_without_repetition'] = data.apply(lambda x: remove_repetition(x.disfluency_text,
                                                                               x.alignments,
                                                                               x.all_errors), axis=1)


def remove_retracing(pred, alignment, all_errors):
    if all_errors is None or not all_errors:
        if pred is not None:
            pred = pred.split()
            transcript = " ".join([w for w in pred[::2]])
            transcript = ""
            return transcript
        else:
            return None
    if pred is None or not pred:
        return None
    pred = pred.split()
    pred_tags = pred[1::2]
    pred_words = pred[::2]
    detected_wret_errors = {}
    detected_wret_idx = {}
    detected_pret_errors = {}
    detected_pret_idx = {}
    _, _, target_wrets, target_prets = all_errors
    for (idx, (word, orig_idx), tag) in zip(list(range(len(alignment))), alignment, pred_tags):
        if tag != "E":
            continue
        for wrep in target_wrets:
            if orig_idx in wrep:
                key = "_".join([str(wi) for wi in wrep])
                val = [True if wi == orig_idx else False for wi in wrep]
                val2 = [idx if wi == orig_idx else None for wi in wrep]
                if key not in detected_wret_errors:
                    detected_wret_errors[key] = val
                    detected_wret_idx[key] = val2
                else:
                    value = detected_wret_errors[key]
                    value2 = detected_wret_idx[key]
                    value = [new or old for new, old in zip(val, value)]
                    value2 = [update(new, old) for new, old in zip(val2, value2)]
                    detected_wret_errors[key] = value
                    detected_wret_idx[key] = value2
        for prep in target_prets:
            if orig_idx in prep:
                key = "_".join([str(wi) for wi in prep])
                val = [True if wi == orig_idx else False for wi in prep]
                val2 = [idx if wi == orig_idx else None for wi in prep]
                if key not in detected_pret_errors:
                    detected_pret_errors[key] = val
                    detected_pret_idx[key] = val2
                else:
                    value = detected_pret_errors[key]
                    value2 = detected_pret_idx[key]
                    value = [new or old for new, old in zip(val, value)]
                    value2 = [update(new, old) for new, old in zip(val2, value2)]
                    detected_pret_errors[key] = value
                    detected_pret_idx[key] = value2
    success_wrep = []
    success_prep = []
    for key, value in detected_wret_errors.items():
        if sum(value) == len(value):
            success_wrep.append(key)
    for key, value in detected_pret_errors.items():
        if sum(value) == len(value):
            success_prep.append(key)
    for key in success_wrep:
        words_to_remove = detected_wret_idx[key]
        for wi in words_to_remove:
            pred_words[wi] = "[TO BE REMOVED]"
    for key in success_prep:
        words_to_remove = detected_pret_idx[key]
        for wi in words_to_remove:
            pred_words[wi] = "[TO BE REMOVED]"
    transcript = " ".join([w for w in pred_words if w != '[TO BE REMOVED]'])
    return transcript


data['transcript_without_retracing'] = data.apply(lambda x: remove_retracing(x.disfluency_text,
                                                                             x.alignments,
                                                                             x.all_errors), axis=1)


def remove_duplicates(text):
    text = [i[0] for i in groupby(text.split())]
    text = " ".join(text)
    return text


def remove_disfluency(text):
    if text is None:
        return
    text = text.split()
    text_tags = text[1::2]
    text_words = text[::2]
    text = ""
    for w, t in zip(text_words, text_tags):
        if t == "E":
            continue
        text += " " + w
    return text


# def remove_uh(text):
#     if text is None:
#         return
#     text = re.sub(r"\b[uhmo]{2,}\b", "", text)
#     text = re.sub(r"\s+", " ", text).strip()
#     text = remove_duplicates(text)
#     text = re.sub(r"\s+", " ", text).strip()
#     return text


data['transcript_without_disfluency'] = data.disfluency_text.apply(lambda x: remove_disfluency(x))
# data['transcript_without_uh'] = data.transcript_without_disfluency.apply(lambda x: remove_uh(x))

data_final = {}
for (idx, (speaker,
           transcript_without_tags,
           transcript_without_disfluency,
           transcript_without_underscore,
           transcript_without_repetition,
           transcript_without_retracing,
           dx,
           mmse)) in data[['speaker',
                           'transcript_without_tags',
                           'transcript_without_disfluency',
                           'transcript_without_underscore',
                           'transcript_without_repetition',
                           'transcript_without_retracing',
                           'dx',
                           'mmse']].iterrows():
    if speaker not in data_final:
        data_final[speaker] = (speaker,
                               transcript_without_tags,
                               transcript_without_disfluency,
                               transcript_without_underscore,
                               transcript_without_repetition,
                               transcript_without_retracing,
                               dx,
                               mmse)
    else:
        _transcript_without_tags = data_final[speaker][1]
        _transcript_without_disfluency = data_final[speaker][2]
        _transcript_without_underscore = data_final[speaker][3]
        _transcript_without_repetition = data_final[speaker][4]
        _transcript_without_retracing = data_final[speaker][5]
        if transcript_without_tags is not None:
            _transcript_without_tags += " " + transcript_without_tags
        if transcript_without_disfluency is not None:
            _transcript_without_disfluency += " " + transcript_without_disfluency
        if transcript_without_underscore is not None:
            _transcript_without_underscore += " " + transcript_without_underscore
        if transcript_without_repetition is not None:
            _transcript_without_repetition += " " + transcript_without_repetition
        if transcript_without_retracing is not None:
            _transcript_without_retracing += " " + transcript_without_retracing
        data_final[speaker] = (speaker,
                               _transcript_without_tags,
                               _transcript_without_disfluency,
                               _transcript_without_underscore,
                               _transcript_without_repetition,
                               _transcript_without_retracing,
                               dx,
                               mmse)

data_final = pd.DataFrame(list(data_final.values()), columns=['speaker',
                                                              'transcript_without_tags',
                                                              'transcript_without_disfluency',
                                                              'transcript_without_underscore',
                                                              'transcript_without_repetition',
                                                              'transcript_without_retracing',
                                                              'dx',
                                                              'mmse'])
data_final.to_pickle('data2020fisher_all.pickle')
data_final.to_csv('data2020fisher_all.csv')
data[['speaker',
      'transcript_without_tags',
      'transcript_without_disfluency',
      'transcript_without_underscore',
      'transcript_without_repetition',
      'transcript_without_retracing',
      'dx',
      'mmse']].to_pickle("data2020fisher_utt_all.pickle")
print()

#
# def count_words(text):
#     if text is None:
#         return
#     text = text.split()
#     return len(text)
#
#
# data_final['total_words'] = data_final.transcript_without_tags.apply(lambda x: count_words(x))
# data_final['num_words_rep'] = data_final.transcript_without_repetition.apply(lambda x: count_words(x))
# data_final['num_words_ret'] = data_final.transcript_without_retracing.apply(lambda x: count_words(x))
# data_final['num_words_both'] = data_final.transcript_without_disfluency.apply(lambda x: count_words(x))
#
#
# def get_percentage(num_words, total_words):
#     if total_words is None:
#         return -1
#     if num_words is None:
#         num_words = 0
#     return (total_words - num_words) / total_words
#
#
# data_final['rep_del'] = data_final.apply(lambda x: get_percentage(x.num_words_rep, x.total_words), axis=1)
# data_final['ret_del'] = data_final.apply(lambda x: get_percentage(x.num_words_ret, x.total_words), axis=1)
# data_final['both_del'] = data_final.apply(lambda x: get_percentage(x.num_words_both, x.total_words), axis=1)
# # data_final.to_pickle('data2020fisher_all.pickle')
# # data[['speaker',
# #       'transcript_without_tags',
# #       'transcript_without_disfluency',
# #       'dx',
# #       'mmse']].to_pickle("data2020fisher_utt_all.pickle")
# # print()
# # total_words = sum(data.total_words.tolist())
# # num_words_rep = sum(data.num_words_rep.tolist())
# # num_words_ret = sum(data.num_words_ret.tolist())
# # num_words_both = sum(data.num_words_both.tolist())
# #
# # print("Rep: ", (total_words - num_words_rep) / total_words)
# # print("Ret: ", (total_words - num_words_ret) / total_words)
# # print("Both: ", (total_words - num_words_both) / total_words)
# import numpy as np
#
# rep_del = [val for val in data_final.rep_del.tolist() if val != -1]
# ret_del = [val for val in data_final.ret_del.tolist() if val != -1]
# both_del = [val for val in data_final.both_del.tolist() if val != -1]
# avg_rep_del = np.mean(rep_del)
# avg_ret_del = np.mean(ret_del)
# avg_both_del = np.mean(both_del)
#
# print("Rep: ", avg_rep_del)
# print("Ret: ", avg_ret_del)
# print("Both: ", avg_both_del)
# for (idx, (text1, text2, text3, text4, text5)) in data[['transcript_without_tags',
#                                                         'transcript_without_disfluency',
#                                                         'transcript_without_uh',
#                                                         'transcript_without_repetition',
#                                                         'transcript_without_retracing']].iterrows():
#     if text2 != text3:
#         print(f'"{text2}"', "\n", f'"{text3}"', "\n\n")
