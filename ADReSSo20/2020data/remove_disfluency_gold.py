import re

import pandas as pd

df = pd.read_csv('data.csv', usecols=['ID', 'mmse'])
id_to_mmse = df.set_index('ID')['mmse'].to_dict()

data = pd.read_pickle('../transcript_with_disfluency_parse_train.pickle')

data['mmse'] = data.speaker.apply(lambda x: id_to_mmse.get(x, None))


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z' ]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


data['transcript_without_tags'] = data.transcript_without_tags.apply(lambda x: clean_text(x))


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
    if '+' in text:
        if text.startswith('+'):
            text = text.replace('+"', '')
            text = text.replace('+', '')
        else:
            text = text.split('+')[0]
    if text.startswith('.'):
        text = text[1:]
    text = re.sub(r"[=:][a-z]+", "", text)
    text = re.sub(r"\[x[0-9 ]+\]", "", text)
    text = text.replace('[]', '')
    text = text.replace('_', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    text_tokens = text.split()
    text_tokens = [t if "@" not in t else "" for t in text_tokens]
    text = " ".join(text_tokens)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def remove_repetition(transcript):
    transcript_tokens = re.split("\s", transcript)
    phrase_repeat_errors = []
    word_repeat_errors = []
    for idx, token in enumerate(transcript_tokens):
        if token == "[/]":
            if idx < 2:
                continue
            if ">" in transcript_tokens[idx - 1]:
                error_i = [idx - 1]
                back_idx = idx - 1
                while back_idx > 0:
                    t = transcript_tokens[back_idx]
                    if back_idx not in error_i:
                        error_i = [back_idx] + error_i
                    if "<" in t:
                        break
                    back_idx -= 1
                phrase_repeat_errors.append(error_i)
            else:
                error_i = [idx - 1]
                word_repeat_errors.append(error_i)
                phrase_repeat_errors.append(error_i)
    for wi_list in word_repeat_errors + phrase_repeat_errors:
        for wi in wi_list:
            transcript_tokens[wi] = '[TO BE REMOVED]'
    transcript_tokens = [w for w in transcript_tokens if w != "[TO BE REMOVED]"]
    transcript_without_repetition = " ".join(transcript_tokens)
    transcript_without_repetition = remove_tags(transcript_without_repetition)
    return transcript_without_repetition


data['transcript_without_repetition'] = data.transcript_with_tags.apply(lambda x: remove_repetition(x))


def remove_retracing(transcript):
    transcript_tokens = re.split("\s", transcript)
    phrase_retrace_errors = []
    word_retrace_errors = []
    for idx, token in enumerate(transcript_tokens):
        if token == "[//]":
            if idx < 2:
                continue
            if ">" in transcript_tokens[idx - 1]:
                error_i = [idx - 1]
                back_idx = idx - 1
                while back_idx > 0:
                    t = transcript_tokens[back_idx]
                    if back_idx not in error_i:
                        error_i = [back_idx] + error_i
                    if "<" in t:
                        break
                    back_idx -= 1
                phrase_retrace_errors.append(error_i)
            else:
                error_i = [idx - 1]
                word_retrace_errors.append(error_i)
                phrase_retrace_errors.append(error_i)
    for wi_list in word_retrace_errors + phrase_retrace_errors:
        for wi in wi_list:
            transcript_tokens[wi] = '[TO BE REMOVED]'
    transcript_tokens = [w for w in transcript_tokens if w != "[TO BE REMOVED]"]
    transcript_without_retrace = " ".join(transcript_tokens)
    transcript_without_retrace = remove_tags(transcript_without_retrace)
    return transcript_without_retrace


data['transcript_without_retracing'] = data.transcript_with_tags.apply(lambda x: remove_retracing(x))

data_final = {}
for (idx, (speaker,
           transcript_without_tags,
           transcript_without_repetition,
           transcript_without_retracing,
           dx,
           mmse)) in data[['speaker',
                           'transcript_without_tags',
                           'transcript_without_repetition',
                           'transcript_without_retracing',
                           'dx',
                           'mmse']].iterrows():
    if speaker not in data_final:
        data_final[speaker] = (speaker,
                               transcript_without_tags,
                               transcript_without_repetition,
                               transcript_without_retracing,
                               dx,
                               mmse)
    else:
        _transcript_without_tags = data_final[speaker][1]
        _transcript_without_repetition = data_final[speaker][2]
        _transcript_without_retracing = data_final[speaker][3]
        if transcript_without_tags is not None:
            _transcript_without_tags += " " + transcript_without_tags
        if transcript_without_repetition is not None:
            _transcript_without_repetition += " " + transcript_without_repetition
        if transcript_without_retracing is not None:
            _transcript_without_retracing += " " + transcript_without_retracing
        data_final[speaker] = (speaker,
                               _transcript_without_tags,
                               _transcript_without_repetition,
                               _transcript_without_retracing,
                               dx,
                               mmse)

data_final = pd.DataFrame(list(data_final.values()), columns=['speaker',
                                                              'transcript_without_tags',
                                                              'transcript_without_repetition',
                                                              'transcript_without_retracing',
                                                              'dx',
                                                              'mmse'])
data.to_pickle("data2020gold.pickle")
# data_a.to_pickle("data_a.pickle")
# data_b.to_pickle("data_b.pickle")
# data_c.to_pickle("data_c.pickle")
print()
