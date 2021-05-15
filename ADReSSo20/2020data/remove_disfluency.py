import re

import pandas as pd

df = pd.read_csv('data.csv', usecols=['ID', 'mmse'])
id_to_mmse = df.set_index('ID')['mmse'].to_dict()

data = pd.read_pickle('../transcript_with_disfluency_parse_train.pickle')

data['mmse'] = data.speaker.apply(lambda x: id_to_mmse.get(x, None))


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z' ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


data['transcript_without_tags'] = data.transcript_without_tags.apply(lambda x: clean_text(x))
data_a = {}
for (idx, (speaker, transcript, dx, mmse)) in data[['speaker',
                                                    'transcript_without_tags',
                                                    'dx',
                                                    'mmse']].iterrows():
    if speaker not in data_a:
        data_a[speaker] = (speaker, transcript, dx, mmse)
    else:
        speaker_transcript = data_a[speaker][1]
        if transcript is not None:
            speaker_transcript += ". " + transcript
        data_a[speaker] = (speaker, speaker_transcript, dx, mmse)

data_a = pd.DataFrame(list(data_a.values()), columns=['speaker', 'transcript', 'dx', 'mmse'])


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
            pred = pred[::2]
            transcript = ""
            for w in pred:
                if w.startswith("'") or w.startswith("n'"):
                    transcript += w
                else:
                    transcript += " " + w
            return transcript.strip()
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
    transcript = ""
    for w in pred_words:
        if w == "[TO BE REMOVED]":
            continue
        if w.startswith("'") or w.startswith("n'"):
            transcript += w
        else:
            transcript += " " + w
    return transcript.strip()


data['transcript_without_repetition'] = data.apply(lambda x: remove_repetition(x.disfluency_text,
                                                                               x.alignments,
                                                                               x.all_errors), axis=1)
data_b = {}
for (idx, (speaker, transcript, dx, mmse)) in data[['speaker',
                                                    'transcript_without_repetition',
                                                    'dx',
                                                    'mmse']].iterrows():
    if speaker not in data_b:
        data_b[speaker] = (speaker, transcript, dx, mmse)
    else:
        speaker_transcript = data_b[speaker][1]
        if transcript is not None:
            speaker_transcript += ". " + transcript
        data_b[speaker] = (speaker, speaker_transcript, dx, mmse)

data_b = pd.DataFrame(list(data_b.values()), columns=['speaker', 'transcript', 'dx', 'mmse'])


def remove_retracing(pred, alignment, all_errors):
    if all_errors is None or not all_errors:
        if pred is not None:
            pred = pred.split()
            pred = pred[::2]
            transcript = ""
            for w in pred:
                if w.startswith("'") or w.startswith("n'"):
                    transcript += w
                else:
                    transcript += " " + w
            return transcript.strip()
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
    transcript = ""
    for w in pred_words:
        if w == "[TO BE REMOVED]":
            continue
        if w.startswith("'") or w.startswith("n'"):
            transcript += w
        else:
            transcript += " " + w
    return transcript.strip()


data['transcript_without_retracing'] = data.apply(lambda x: remove_retracing(x.disfluency_text,
                                                                             x.alignments,
                                                                             x.all_errors), axis=1)
data_c = {}
for (idx, (speaker, transcript, dx, mmse)) in data[['speaker',
                                                    'transcript_without_retracing',
                                                    'dx',
                                                    'mmse']].iterrows():
    if speaker not in data_c:
        data_c[speaker] = (speaker, transcript, dx, mmse)
    else:
        speaker_transcript = data_c[speaker][1]
        if transcript is not None:
            speaker_transcript += ". " + transcript
        data_c[speaker] = (speaker, speaker_transcript, dx, mmse)

data_c = pd.DataFrame(list(data_c.values()), columns=['speaker', 'transcript', 'dx', 'mmse'])

data_a.to_pickle("data_a.pickle")
data_b.to_pickle("data_b.pickle")
data_c.to_pickle("data_c.pickle")
print()
