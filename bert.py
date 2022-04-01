from transformers import BertModel, BertTokenizer
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from stse.bytes import bit_vect


class BertBased(nn.Module):
    def __init__(self, num_classes, bert_config='bert-large-uncased'):
        super(BertBased, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bert = BertModel.from_pretrained(bert_config).to(device)
        # For each word
        self.fc = nn.Linear(1024, num_classes)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.bert(x)['last_hidden_state']
        x = self.fc(x)
        return self.softmax(x)


class BertDataset(Dataset):
    def __init__(self, x, y):
        super(BertDataset, self).__init__()
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        _x = self.x[index]
        _y = self.y[index]
        return _x, _y

if __name__ == '__main__':
    # Define globals
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CONFIG = 'bert-large-uncased'

    # Define BERT model
    bert_model = BertModel.from_pretrained(CONFIG).to(DEVICE)

    # Import data
    notes_df = pd.read_csv('data/patient_notes.csv')
    train_df = pd.read_csv('data/train.csv')
    features_df = pd.read_csv('data/features.csv')

    # One-hot encode features
    features_df['feature_vect'] = [bit_vect(len(features_df) + 1, i) for i in range(len(features_df['feature_text']))]

    # APPEND AND CLEAN DATA
    train_df = train_df[train_df['annotation'] != '[]']  # Drop blank annotations ('[]')
    data = train_df.merge(features_df[['feature_num', 'feature_text', 'feature_vect']], on='feature_num')  # Add features
    data = data.merge(notes_df[['pn_num', 'pn_history']], on='pn_num')  # Add notes
    # seps = [' ', ',', ';', ':', '.', '!', '?', '-', '_', '\n']  # WORRY ABOUT THIS LATER
    word_lists = data['pn_history'].apply(lambda x: np.array(x.split(' '))).to_numpy()  # Convert notes to lists of words
    data = data.dropna().reset_index(drop=True)  # Drop and reindex any leftover trouble-makers

    # build annotations
    # annotations = [i.translate(i.maketrans('', '', '[]\'')).split(' ') for i in data['annotation']]
    # assert len(annotations) == len(word_lists)
                
    # Tokenize word lists and cast to tensors
    tokenizer = BertTokenizer.from_pretrained(CONFIG)
    encoded_word_lists = [tokenizer.encode(x.tolist()) for x in word_lists]
    X = [torch.cuda.ShortTensor(np.array(x).reshape(1, -1)) for x in encoded_word_lists]

    y = [torch.cuda.ByteTensor(x) for x in data['feature_vect'].to_numpy()]


    def training_loop():
        # Epoch loss history for plotting
        train_loss_history = []
        val_loss_history = []
        
        # Loss function, model, and optimizer
        criterion = nn.NLLLoss()
        model = CBOW(vocab=vocab, embedding_dim=embedding_size, context=2*context_size).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


    # a = torch.Tensor(encoded_word_lists)
    # print(torch.tensor(encoded_word_lists[0]))
    # fits = [model(torch.cuda.IntTensor(np.array(x).reshape(1, -1))) for x in encoded_word_lists]
    # out = []
    # for x in [encoded_word_lists[0]]:
    #     out = bert_model(torch.cuda.IntTensor(np.array(x).reshape(1, -1)).detach())

    # print(out['last_hidden_state'].shape)
    # print(len(tokenizer.decode(encoded_word_lists[0])))
    # print('\n\n', len(word_lists[0]))

        # print(ape)
    # a = model(torch.tensor(b))  # Shape: batch_size, seq_length
    # print(a)
    # print(fits)
    # print(fits)


    # annotations = [i.translate(i.maketrans('', '', '[]\'')).split(' ') for i in train_df['annotation']]
    #     # print(len(i.split(' ')))
        
    # # print(word_lists[0])
    # print(annotations[0])
    # print(word_lists[0 + 16])
    # for word in word_lists[0]:
    #     print(word if word in annotations[0] else 'NONE')