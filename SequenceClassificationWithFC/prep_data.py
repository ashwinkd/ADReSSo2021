from global_vars import *


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
                pause_num = re.sub('[^0-9]', '', surrounding_speech)
                pause_token = "[P{}]".format(pause_num)
                pause_token = UNUSED_TOKEN[pause_token]
                tokens.append(pause_token)
                print()
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    return tokenizer.convert_tokens_to_ids(tokens)


def add_padding(input_word_ids):
    input_type = []
    for idx, embedding in enumerate(input_word_ids):
        embedding_len = len(embedding)
        e_input_type = np.ones(embedding_len, dtype=np.int64).tolist()
        if embedding_len < max_len_embedding:
            zeros = np.zeros(max_len_embedding - embedding_len, dtype=np.int64).tolist()
            e_input_type += zeros
            embedding += zeros
        input_type.append(e_input_type)
        input_word_ids[idx] = embedding
    return {INPUT_IDS: input_word_ids, ATTENTION_MASK: input_type}


def bert_encode(transcripts, tokenizer):
    input_word_ids = [encode_sentence(s, tokenizer)
                      for s in transcripts]
    input_word_ids = add_padding(input_word_ids)
    return input_word_ids


## Data
data = pd.read_pickle('data.pickle')
data = pd.get_dummies(data, columns=['dx'])

X_transcript = data.transcript.to_numpy()
# X_0_500 = data['0-500'].to_numpy()
# X_500_1000 = data['500-1000'].to_numpy()
# X_1000_2000 = data['1000-2000'].to_numpy()
# X_2000 = data['2000'].to_numpy()
Y = data.dx_cn.to_numpy()

## Tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

X_train, X_test, y_train, y_test = train_test_split(X_transcript, Y, test_size=0.2, random_state=42, stratify=Y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

tokens_train = bert_encode(X_train, tokenizer)
tokens_val = bert_encode(X_val, tokenizer)
tokens_test = bert_encode(X_test, tokenizer)

# for train set
train_seq = torch.tensor(tokens_train[INPUT_IDS])
train_mask = torch.tensor(tokens_train[ATTENTION_MASK])
train_y = torch.tensor(y_train.tolist())

# for validation set
val_seq = torch.tensor(tokens_val[INPUT_IDS])
val_mask = torch.tensor(tokens_val[ATTENTION_MASK])
val_y = torch.tensor(y_val.tolist())

# for test set
test_seq = torch.tensor(tokens_test[INPUT_IDS])
test_mask = torch.tensor(tokens_test[ATTENTION_MASK])
test_y = torch.tensor(y_test.tolist())

# define a batch size
batch_size = 4

# wrap tensors
train_data = TensorDataset(train_seq, train_mask, train_y)

# sampler for sampling the data during training
train_sampler = RandomSampler(train_data)

# dataLoader for train set
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# wrap tensors
val_data = TensorDataset(val_seq, val_mask, val_y)

# sampler for sampling the data during training
val_sampler = SequentialSampler(val_data)

# dataLoader for validation set
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)
