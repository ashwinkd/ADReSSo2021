from Data.Prep.global_vars import *

transcript_dir = '../Dataset/penn_files/transcripts'
pause_dir = '../Dataset/penn_files/pause_features'
data = pd.read_csv(os.path.join(adresso_data_directory, 'adresso-train-mmse-scores.csv'),
                   usecols=["adressfname", "mmse", "dx"])
data['transcript'] = [dict() for _ in range(len(data))]
data['0-500'] = 0
data['500-1000'] = 0
data['1000-2000'] = 0
data['2000'] = 0

data_dict = data.set_index('adressfname').T.to_dict()


def get_str(transcript):
    speaker_transcript = []
    keys = sorted(transcript.keys())
    for key in keys:
        _, _, segment_transcript = transcript[key]
        speaker_transcript += [token
                               if re.findall('\[P\d\]', token)
                               else token.lower()
                               for token in segment_transcript]
    if speaker_transcript:
        return " ".join(speaker_transcript)
    return


def get_transcript(filepath):
    with open(filepath, 'r') as fptr:
        data = fptr.read()
    return data


def get_pause_features(pause_filepath):
    try:
        data = json.load(open(pause_filepath, 'r'))
        return data['0-500'], data['500-1000'], data['1000-2000'], data['2000']
    except Exception as e:
        print(e)
        return 0, 0, 0, 0


for file in os.listdir(transcript_dir):
    transcript_filepath = os.path.join(transcript_dir, file)
    pause_filepath = os.path.join(pause_dir, file.replace('.txt', '.json'))
    speaker_id, begin, end = file.split('.')[0].split('_')
    begin = int(begin)
    end = int(end)
    segment_transcript = get_transcript(transcript_filepath)
    _0_500, _500_1000, _1000_2000, _2000 = get_pause_features(pause_filepath)
    if speaker_id not in data_dict:
        continue
    data_dict[speaker_id]['transcript'][begin] = (begin, end, segment_transcript.split())
    data_dict[speaker_id]['0-500'] += _0_500
    data_dict[speaker_id]['500-1000'] += _500_1000
    data_dict[speaker_id]['1000-2000'] += _1000_2000
    data_dict[speaker_id]['2000'] += _2000

data = pd.concat([pd.DataFrame(list(data_dict.keys()),
                               columns=['adressfname']),
                  pd.DataFrame(list(data_dict.values()))], axis=1)
data['transcript'] = data['transcript'].apply(lambda x: get_str(x))
print()
data.dropna().to_pickle(data_directory)
