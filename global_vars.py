import os
import pandas as pd
import re
import json

import os

import numpy as np

import tensorflow_datasets as tfds
tfds.disable_progress_bar()

from transformers import AutoModel, BertTokenizerFast
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
# optimizer from hugging face transformers
from transformers import AdamW

from sklearn.utils.class_weight import compute_class_weight


data_directory = "data.pickle"

device = torch.device("cuda")

INPUT_IDS = 'input_ids'
ATTENTION_MASK = 'attention_mask'

gs_folder_bert = "gs://cloud-tpu-checkpoints/bert/v3/uncased_L-12_H-768_A-12"
adresso_data_directory = '/mnt/f/Research/ADReSSo/ADReSSo21/diagnosis/train'

UNUSED_TOKEN = {'[P1]': '[unused0]',
                '[P2]': '[unused1]',
                '[P3]': '[unused2]'}

max_len_embedding = 500
