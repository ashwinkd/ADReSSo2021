import os
import re

import pandas as pd

in_directory = "/mnt/f/Research/ADReSSo/2020/train/transcription"
out_directory = "train/transcription"
speaker_dict = {}
df = pd.DataFrame(columns=['speaker', 'transcript_without_tags', 'transcript_with_tags', 'dx'])
all_lines = ""


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
    text = text.replace("...", '')
    text = text.replace("..", '')
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
    if '+' in text:
        if text.startswith('+'):
            text = text.replace('+"', '')
            text = text.replace('+', '')
        else:
            text = text.split('+')[0]
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


for cat in ["Control", "Dementia"]:
    for file in os.listdir(os.path.join(in_directory, cat)):
        in_filepath = os.path.join(in_directory, cat, file)
        speaker = file.split('.')[0]
        out_filepath = os.path.join(out_directory, cat, f"{speaker}.txt")
        ##############
        if speaker not in speaker_dict:
            speaker_dict[speaker] = []
        lines = []
        with open(in_filepath, 'r') as ifptr:
            lines = ifptr.readlines()
        idx = 0
        while idx < len(lines):
            line = lines[idx]
            if line.startswith("*PAR"):
                par_line = line
                idx2 = idx + 1
                while idx2 < len(lines):
                    next_line = lines[idx2]
                    if re.match(r"\d+\_\d+", next_line):
                        par_line += " " + next_line
                        break
                    if ":\t" in next_line:
                        break
                    par_line += " " + next_line
                    idx2 += 1
                par_line_without_tags = remove_tags(par_line)
                print(par_line_without_tags)
                all_lines += par_line_without_tags + "\n"
                df = df.append({'speaker': speaker,
                                'transcript_without_tags': par_line_without_tags,
                                'transcript_with_tags': par_line,
                                "dx": cat},
                               ignore_index=True)
            idx += 1
df.to_pickle("transcripts.pickle")
with open("all_text.txt", 'w') as fptr:
    fptr.write(all_lines)
    fptr.close()
