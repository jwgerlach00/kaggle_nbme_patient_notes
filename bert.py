from __future__ import annotations
from pyparsing import anyOpenTag
from transformers import BertModel, BertTokenizer
import numpy as np
import pandas as pd
import torch
from stse.bytes import bit_vect


config = 'bert-large-uncased'
model = BertModel.from_pretrained(config)

# print(torch.Tensor(np.array(['sdfs'])))

# print(model(torch.tensor(['sdsdf'])))

# Import data
notes_df = pd.read_csv('data/patient_notes.csv')
train_df = pd.read_csv('data/train.csv')
features_df = pd.read_csv('data/features.csv')

# One-hot encode features
features_df['feature_vect'] = [bit_vect(len(features_df) + 1, i) for i in range(len(features_df['feature_text']))]

# APPEND AND CLEAN DATA
# Drop blank annotations ('[]')
train_df = train_df[train_df['annotation'] != '[]']

# Add features
data = train_df.merge(features_df[['feature_num', 'feature_text', 'feature_vect']], on='feature_num')

# Add notes
data = data.merge(notes_df[['pn_num', 'pn_history']], on='pn_num')

# seps = [' ', ',', ';', ':', '.', '!', '?', '-', '_', '\n']  # WORRY ABOUT THIS LATER
# Convert notes to lists of words
word_lists = data['pn_history'].apply(lambda x: np.array(x.split(' '))).to_numpy()


# Drop and reindex any residuals
data = data.dropna().reset_index(drop=True)

annotations = [i.translate(i.maketrans('', '', '[]\'')).split(' ') for i in data['annotation']]

assert len(annotations) == len(word_lists)

for note, ann in zip([word_lists[0]], [annotations[0]]):
    for word in note:
        if word in ann:
            print(word)
            
# Tokenize word lists
tokenizer = BertTokenizer.from_pretrained(config)
encoded_word_lists = [tokenizer.encode(x.tolist()) for x in word_lists]


# a = torch.Tensor(encoded_word_lists)
# print(torch.tensor(encoded_word_lists[0]))
fits = [model(torch.tensor(np.array(x).reshape(1, -1))) for x in encoded_word_lists]
# a = model(torch.tensor(b))  # Shape: batch_size, seq_length
# print(a)
print(fits)


# annotations = [i.translate(i.maketrans('', '', '[]\'')).split(' ') for i in train_df['annotation']]
#     # print(len(i.split(' ')))
    
# # print(word_lists[0])
# print(annotations[0])
# print(word_lists[0 + 16])
# for word in word_lists[0]:
#     print(word if word in annotations[0] else 'NONE')