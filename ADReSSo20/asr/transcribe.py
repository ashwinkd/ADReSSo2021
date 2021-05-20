import os
import re

import pandas as pd
from google.cloud import speech
from pydub import AudioSegment

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/mnt/c/adresso-asr-dab9362daad1.json"
import io
from ADReSSo20.disfluency.remove_tags import remove_tags
from ADReSSo20.disfluency.get_accuracy import resolve_text

sample_rate = 16000
directory = "/mnt/f/Research/ADReSSo/2020/train/"


def read_data(transcript_filepath):
    with open(transcript_filepath, 'r') as read_fptr:
        text = read_fptr.read()
        read_fptr.close()
    text_lines = text.split("\n")
    return text_lines


def get_utterance(transcript_filepath):
    lines = read_data(transcript_filepath)
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        segment = None
        if not line.startswith("*PAR:\t"):
            idx += 1
            continue
        forward_idx = idx
        while forward_idx < len(lines):
            forward_line = lines[forward_idx]
            if not line.endswith(forward_line):
                line += " " + forward_line
            if re.search(r"\d+\_\d+", forward_line):
                for seg in re.findall(r"\d+\_\d+", forward_line):
                    start, end = seg.split("_")
                    start = int(start)
                    end = int(end)
                    segment = (start, end)
                    break
                idx = forward_idx + 1
                break
            forward_idx += 1
        line = remove_tags(line)
        yield (line, segment)


def transcribe(speech_file):
    full_result = ""
    # Transcribe the given audio file

    client = speech.SpeechClient()

    with io.open(speech_file, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                                      sample_rate_hertz=sample_rate,
                                      language_code="en-US",
                                      enable_automatic_punctuation=True,
                                      use_enhanced=True,
                                      model="phone_call",
                                      )

    response = client.recognize(config=config, audio=audio)

    # Each result is for a consecutive portion of the audio. Iterate through
    # them to get the transcripts for the entire audio file.
    for result in response.results:
        # The first alternative is the most likely one for this portion.
        full_result += result.alternatives[0].transcript
        # print(u"Transcript: {}".format(result.alternatives[0].transcript))
    return full_result


def get_audio_segment(speaker, segment, audio_filepath):
    start, end = segment
    audio_file = AudioSegment.from_wav(audio_filepath)
    audio_file = audio_file.set_frame_rate(sample_rate)
    segment_path = './audio/{}_{}_{}.wav'.format(speaker, start, end)
    audio_segment = audio_file[start:end]
    audio_segment.export(segment_path, format="wav")
    return segment_path


def get_data(speaker, transcript_filepath, audio_filepath):
    for participant_utterance, segment in get_utterance(transcript_filepath):
        utt_id = f"{speaker}_{segment[0]}_{segment[1]}"
        segment_filepath = get_audio_segment(speaker, segment, audio_filepath)
        asr_utterance = transcribe(segment_filepath)
        yield [utt_id, participant_utterance, asr_utterance, segment_filepath]


def main():
    columns = ['utt_id', 'gold_utterance', 'asr_utterance', 'audio_filepath']
    df = pd.DataFrame(columns=columns)
    for cat1, cat2 in zip(["cc", "cd"], ["Control", "Dementia"]):
        cat_path = os.path.join(directory, 'transcription', cat2)
        audio_path = os.path.join(directory, 'Full_wave_enhanced_audio', cat1)
        for file in os.listdir(cat_path):
            transcript_filepath = os.path.join(cat_path, file)
            speaker, _ = file.split('.')
            audio_filepath = os.path.join(audio_path, f"{speaker}.wav")
            for data in get_data(speaker, transcript_filepath, audio_filepath):
                row = dict(zip(columns, data))
                df = df.append(row, ignore_index=True)
        df.to_pickle('adresso2020_transcripts.pickle')


if __name__ == '__main__':
    main()
