import os

import pandas as pd

stats = pd.read_csv('./stats.csv')
stats['num_utt'] = None
trdirectory_cc = "/mnt/f/Research/ADReSSo/2020/train/transcription/Control"
trdirectory_cd = "/mnt/f/Research/ADReSSo/2020/train/transcription/Dementia"
tsdirectory = "/mnt/f/Research/ADReSSo/2020/test/transcription"


def read_data(transcript_filepath):
    with open(transcript_filepath, 'r') as read_fptr:
        text = read_fptr.read()
        read_fptr.close()
    text_lines = text.split("\n")
    return text_lines


def get_num_utt(filepath):
    lines = read_data(filepath)
    num_utt = 0
    for line in lines:
        if line.startswith("*PAR:\t"):
            num_utt += 1
    return num_utt


for dir in [trdirectory_cc, trdirectory_cd, tsdirectory]:
    for file in os.listdir(dir):
        speaker, _ = file.split('.')
        filepath = os.path.join(dir, file)
        num_utt = get_num_utt(filepath)
        stats.loc[(stats.speaker == speaker), 'num_utt'] = num_utt

stats = stats[['speaker', 'num_utt', 'age', 'gender', 'mmse', 'dx', 'set']]
stats.to_pickle('stats.pickle')
stats.to_csv('stats.csv')
