import re

import pandas as pd

dys_and_parse = pd.read_pickle('all_tags.pkl')
idx = 0
keys = set()


def get_key(text):
    text = text.split()
    twords = text[::2]
    text = ""
    for t in twords:
        t = normalize(t)
        if t.startswith("'") or t.startswith("n'"):
            text += t
        else:
            text += " " + t
    return text.strip()


def normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-z' ]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# parse_tree = open('adresso20_text_dys.txt', 'r').readlines()
dys_and_parse['utterance'] = dys_and_parse.disfluency.apply(lambda x: get_key(x))
dys_and_parse = dys_and_parse.set_index('utterance').T.to_dict('list')

data = pd.read_pickle('transcript_with_disfluency_parse.pickle')


def get_disfluency(utterance):
    utterance_normal = normalize(utterance)
    try:
        _, disfluency = dys_and_parse[utterance_normal]
        keys.add(utterance_normal)
        return disfluency
    except:
        if utterance:
            print(utterance, utterance_normal)
        return None


def get_parse(utterance):
    utterance_normal = normalize(utterance)
    try:
        parse, _ = dys_and_parse[utterance_normal]
        return parse
    except:
        return None


data["disfluency_text"] = data.transcript_without_tags.apply(lambda x: get_disfluency(x))
data["parse_tree"] = data.transcript_without_tags.apply(lambda x: get_parse(x))

data[['speaker',
      'utt_id',
      'dx',
      'transcript_without_tags',
      'transcript_with_tags',
      'disfluency_text',
      'parse_tree',
      'recall']].to_pickle('transcript_with_disfluency_parse.pickle')
print("#" * 20)
for k in keys:
    dys_and_parse.pop(k)
for (key, (parse, dtext)) in dys_and_parse.items():
    print(dtext)
