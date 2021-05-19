import random
import re

import numpy as np
import pandas as pd
import torch
from model_choice import model_name
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset, random_split

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
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_basic_tokenize=False)
    return tokenizer


def get_roberta_tokenizer():
    from transformers import RobertaTokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    return tokenizer


tokenizer = locals()[f"get_{model_name.lower()}_tokenizer"]()


def encode_sentence(transcript, tokenizer, include_pauses=True):
    if include_pauses:
        transcript = re.sub(r"\[P1\]", "[unused0]", transcript)
        transcript = re.sub(r"\[P2\]", "[unused1]", transcript)
        transcript = re.sub(r"\[P3\]", "[unused2]", transcript)
    else:
        transcript = re.sub(r"\[P\d\]", "", transcript)
    tokens = list(tokenizer.tokenize(transcript))
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


def bert_encode(transcripts, tokenizer, include_pauses=True):
    input_word_ids = [encode_sentence(s, tokenizer, include_pauses)
                      for s in transcripts]
    input_word_ids = add_padding(input_word_ids)
    return input_word_ids


tokens_train = bert_encode(train_text.to_list(), tokenizer)
tokens_test = bert_encode(temp_text.to_list(), tokenizer)

_tokens_train = bert_encode(train_text.to_list(), tokenizer, False)
_tokens_test = bert_encode(temp_text.to_list(), tokenizer, False)

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

encode_sentence("the water is running", tokenizer)
