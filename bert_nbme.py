from transformers import BertModel
import torch
from torch import nn
from torch.utils.data import Dataset


class BertNN(nn.Module):
    def __init__(self, num_classes, bert_config='bert-base-uncased', bert_hidden_size=768):
        super(BertNN, self).__init__()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.bert = BertModel.from_pretrained(bert_config).to(device)
        
        # For each word
        self.fc = nn.Linear(bert_hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.bert(x)['last_hidden_state']
        x = self.fc(x)
        return self.softmax(x).squeeze()


class BertNNDataset(Dataset):
    def __init__(self, x, y):
        super(BertNNDataset, self).__init__()
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        _x = self.x[index]
        _y = self.y[index]
        return _x, _y
