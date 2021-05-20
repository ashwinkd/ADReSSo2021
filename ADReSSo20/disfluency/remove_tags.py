import os
import re

import pandas as pd


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


def get_id(text):
    for m in re.findall(r"\d+\_\d+", text):
        return m


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z' ]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def count_disfluencies(text):
    word_repetition = 0
    phrase_repetition = 0
    word_retrace = 0
    phrase_retrace = 0
    for m in re.finditer(r"\[\/\]", text):
        start_rep, end_rep = m.start(0), m.end(0)
        if text[start_rep - 2] == '>':
            phrase_repetition += 1
        else:
            word_repetition += 1
    for m in re.finditer(r"\[\/\/\]", text):
        start_rep, end_rep = m.start(0), m.end(0)
        if text[start_rep - 2] == '>':
            phrase_retrace += 1
        else:
            word_retrace += 1
    polysyllabic_word_count = len(re.findall(r"\[x\s\d+\]", text))
    return [polysyllabic_word_count, word_repetition, phrase_repetition, word_retrace, phrase_retrace]


def add_to_class(errors, cat, classwise_utterances, classwise_error):
    classwise_utterances[cat] += 1
    polysyllabic_word_count, word_repetition, phrase_repetition, word_retrace, phrase_retrace = errors
    classwise_error[cat]['polysyllabic_word_count'] += polysyllabic_word_count
    classwise_error[cat]['word_repetition'] += word_repetition
    classwise_error[cat]['phrase_repetition'] += phrase_repetition
    classwise_error[cat]['word_retrace'] += word_retrace
    classwise_error[cat]['phrase_retrace'] += phrase_retrace
    return classwise_utterances, classwise_error


def main():
    in_directory = "/mnt/f/Research/ADReSSo/2020/"
    out_directory = "train/transcription"
    speaker_dict = {}
    columns = ['speaker',
               'utt_id',
               'dx',
               'transcript_without_tags',
               'transcript_with_tags',
               'polysyllabic_word_count',
               'word_repetition',
               'phrase_repetition',
               'word_retrace',
               'phrase_retrace']
    df = pd.DataFrame(columns=columns)
    all_lines = ""

    classwise_error = {"Control": {'polysyllabic_word_count': 0,
                                   'word_repetition': 0,
                                   'phrase_repetition': 0,
                                   'word_retrace': 0,
                                   'phrase_retrace': 0},
                       "Dementia": {'polysyllabic_word_count': 0,
                                    'word_repetition': 0,
                                    'phrase_repetition': 0,
                                    'word_retrace': 0,
                                    'phrase_retrace': 0}
                       }

    classwise_utterances = {"Control": 0,
                            "Dementia": 0
                            }

    for phase in ["train", "test"]:
        for cat in ["Control", "Dementia"]:
            if phase == "train":
                filepath = os.path.join(in_directory, phase, 'transcription', cat)
            else:
                filepath = os.path.join(in_directory, phase, 'transcription')
            for file in os.listdir(filepath):
                in_filepath = os.path.join(filepath, file)
                speaker = file.split('.')[0]

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
                        par_line_without_tags = clean_text(par_line_without_tags)
                        utt_id = get_id(par_line)
                        errors = count_disfluencies(par_line)
                        if phase == "train":
                            classwise_utterances, classwise_error = add_to_class(errors,
                                                                                 cat,
                                                                                 classwise_utterances,
                                                                                 classwise_error)
                        print(par_line_without_tags)
                        all_lines += f"{par_line_without_tags}\n"
                        row = dict(zip(columns, [speaker,
                                                 utt_id,
                                                 cat if phase == "train" else None,
                                                 par_line_without_tags,
                                                 par_line
                                                 ] + errors))
                        df = df.append(row, ignore_index=True)
                    idx += 1
    df.to_pickle('transcripts.pickle')
    print(classwise_error)
    for cat, error_dict in classwise_error.items():
        utt_count = classwise_utterances[cat]
        for error, num in error_dict.items():
            classwise_error[cat][error] = num / utt_count
    print(classwise_error)
    print(classwise_utterances)
    with open("all_text.txt", 'w') as fptr:
        fptr.write(all_lines)
        fptr.close()


if __name__ == '__main__':
    main()
