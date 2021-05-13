import re

import pandas as pd

dys_and_parse = pd.read_pickle('all_tags.pkl')
idx = 0


def get_key(text):
    text = text.replace("_", "")
    text = text.replace("E", "")
    text = re.sub(r"\s", "", text).strip()
    return text.lower()


# parse_tree = open('adresso20_text_dys.txt', 'r').readlines()
dys_and_parse['key'] = dys_and_parse.disfluency.apply(lambda x: get_key(x))
dys_and_parse = dys_and_parse.set_index('key').T.to_dict('list')

data = pd.read_pickle('transcripts.pickle')


def get_disfluency(utterance):
    ukey = get_key(utterance)
    try:
        _, disfluency = dys_and_parse[ukey]
        return disfluency
    except:
        print(utterance)
        return None


def get_parse(utterance):
    ukey = get_key(utterance)
    try:
        parse, _ = dys_and_parse[ukey]
        return parse
    except:
        print(utterance)
        return None


data["disfluency_text"] = data.transcript_without_tags.apply(lambda x: get_disfluency(x))
data["parse_tree"] = data.transcript_without_tags.apply(lambda x: get_parse(x))


def get_phase(x):
    if x in ["Control", "Dementia"]:
        return "TRAIN"
    else:
        return "TEST"


data['test_or_train'] = data.dx.apply(lambda x: get_phase(x))

data_test = data[data['test_or_train'] == "TEST"]
data_train = data[data['test_or_train'] == "TRAIN"]
data[['speaker',
      'utt_id',
      'dx',
      'transcript_without_tags',
      'transcript_with_tags',
      'disfluency_text',
      'parse_tree']].to_pickle('transcript_with_disfluency_parse.pickle')
