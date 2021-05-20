import pandas as pd

from ADReSSo20.disfluency.get_accuracy import resolve_text


def resolve_fisher(text):
    if text is None or not text:
        return "PLACEHOLDER"
    text = resolve_text(text)
    text = " ".join(text)
    return text


data = pd.read_pickle('./adresso2020_transcripts.pickle')
data.gold_utterance = data.gold_utterance.apply(lambda x: resolve_fisher(x))
data.asr_utterance = data.asr_utterance.apply(lambda x: resolve_fisher(x))
data.to_pickle('./adresso2020_transcripts.pickle')
gold = data.gold_utterance.tolist()
asr = data.asr_utterance.tolist()

gold = "\n".join(gold)
asr = "\n".join(asr)

with open('all_text2020_gold.txt', 'w') as goldfptr:
    goldfptr.write(gold)
    goldfptr.close()
with open('all_text2020_asr.txt', 'w') as asrfptr:
    asrfptr.write(asr)
    asrfptr.close()
