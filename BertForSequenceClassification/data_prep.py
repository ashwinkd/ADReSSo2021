import random
import re

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset, random_split

from model_choice import model_name

#################### Global Variables ####################

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
UNUSED_TOKEN = {1: '[unused0]',
                2: '[unused1]',
                3: '[unused2]'}
to_categorical = {'cn': 0,
                  'ad': 1}
seed_val = 42
batch_size = 32
max_seq_len = 512

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
#################### Read Data ####################

df = pd.read_pickle('data.pickle')
df.dx = df.dx.apply(lambda x: to_categorical[x])

#################### Train and Test Sets ####################

train_text, temp_text, train_labels, temp_labels = train_test_split(df['transcript'], df['dx'],
                                                                    random_state=seed_val,
                                                                    test_size=0.3,
                                                                    stratify=df['dx'])


#################### BERT features ####################

def get_bert_tokenizer():
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return tokenizer


def get_roberta_tokenizer():
    from transformers import RobertaTokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    return tokenizer


tokenizer = locals()[f"get_{model_name.lower()}_tokenizer"]()


def encode_sentence(transcript, tokenizer):
    tokens = []
    continous_speech = re.split(r'\[P\d\]', transcript)
    if not continous_speech:
        transcript = str.encode(transcript, 'utf-8')
        tokens = list(tokenizer.tokenize(transcript))
    else:
        for idx, speech in enumerate(continous_speech):
            tokens += list(tokenizer.tokenize(speech))
            if idx + 1 < len(continous_speech):
                surrounding_speech = r'{}\[P\d\]{}'.format(continous_speech[idx], continous_speech[idx + 1])
                surrounding_speech = re.findall(surrounding_speech, transcript)[0]
                pause_num = int(re.sub('[^0-9]', '', surrounding_speech))
                tokens.append(UNUSED_TOKEN[pause_num])
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    return token_ids


def add_padding(input_word_ids):
    input_type = []
    for idx, embedding in enumerate(input_word_ids):
        embedding_len = len(embedding)
        e_input_type = np.ones(embedding_len, dtype=np.int64).tolist()
        if embedding_len < max_seq_len:
            zeros = np.zeros(max_seq_len - embedding_len, dtype=np.int64).tolist()
            e_input_type += zeros
            embedding += zeros
        elif embedding_len > max_seq_len:
            embedding = embedding[:max_seq_len - 1] + [102]
            e_input_type = e_input_type[:max_seq_len]
        input_type.append(torch.tensor([e_input_type]))
        input_word_ids[idx] = torch.tensor([embedding])
    return {'input_ids': input_word_ids, 'attention_mask': input_type}


def bert_encode(transcripts, tokenizer):
    input_word_ids = [encode_sentence(s, tokenizer)
                      for s in transcripts]
    input_word_ids = add_padding(input_word_ids)
    return input_word_ids


tokens_train = bert_encode(train_text.to_list(), tokenizer)
tokens_test = bert_encode(temp_text.to_list(), tokenizer)

#################### Torch Dataset ####################

input_ids = torch.cat(tokens_train['input_ids'], dim=0)
attention_masks = torch.cat(tokens_train['attention_mask'], dim=0)
labels = torch.tensor(train_labels.to_list())

input_ids_test = torch.cat(tokens_test['input_ids'], dim=0)
attention_masks_test = torch.cat(tokens_test['attention_mask'], dim=0)
labels_test = torch.tensor(temp_labels.to_list())
####################
dataset = TensorDataset(input_ids, attention_masks, labels)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
####################
train_dataloader = DataLoader(
    train_dataset,  # The training samples.
    sampler=RandomSampler(train_dataset),  # Select batches randomly
    batch_size=batch_size  # Trains with this batch size.
)

validation_dataloader = DataLoader(
    val_dataset,  # The validation samples.
    sampler=SequentialSampler(val_dataset),  # Pull out batches sequentially.
    batch_size=batch_size  # Evaluate with this batch size.
)
