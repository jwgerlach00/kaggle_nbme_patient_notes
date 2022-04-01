from re import L
import torch
from torch import nn
from torch.utils.data import Dataset
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer


class RNModel(nn.Module):
    def __init__(self, input_size, rnn_hidden_size, rnn_num_layers, out_size):
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=rnn_hidden_size, num_layers=rnn_num_layers, 
                           batch_first=True)
        self.fc1 = nn.Linear(rnn_hidden_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, out_size)
        
    def forward(self, x):
        x = self.rrn(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.out(x)
        x = F.one_hot(x, num_classes=1)
        
class BertBased(nn.Module):
    def __init__(self, num_classes, bert_config='bert-large-uncased'):
        super(BertBased, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bert = BertModel.from_pretrained(bert_config).to(device)
        # For each word
        self.fc = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        x = self.bert(x)
        return self.fc(x)


class RNNDataset(Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        _x = self.x[index]
        _y = self.y[index]
        return _x, _y

# def attention_layer(dot):
#     F.softmax(dot, dim=2)  # Row-wise self-attention probabilities
    
#     # Batch matrix multiplication
#     out = torch.bmm(dot, values).view(b, h, t, s)
    
#     # swap h, t back, unify heads
#     out = out.transpose(1, 2).contiguous().view(b, t, s * h)

#     return self.unifyheads(out)



# def training_loop():
#     # Generate TRAIN context
#     context_train = []
#     for smiles in encoded_train:
#         single_smiles_data = []
#         for i in range(context_size, len(smiles) - 3):
#             # Hard-coded for context size
#             context = [smiles[i - context_size], smiles[i - context_size + 1], smiles[i - context_size + 2],  # before el
#                        smiles[i + context_size - 2], smiles[i + context_size - 1], smiles[i + context_size]]  # after el
#             target = smiles[i]

#             # Tuple together context and target
#             single_smiles_data.append((context, target))

#         # Append all the data together but preserve the separation of context between smiles
#         context_train += single_smiles_data

#     # Generate VAL context
#     context_val = []
#     for smiles in encoded_val:
#         single_smiles_data = []
#         for i in range(context_size, len(smiles) - 3):
#             # Hard-coded for context size
#             context = [smiles[i - context_size], smiles[i - context_size + 1], smiles[i - context_size + 2],
#                        # before el
#                        smiles[i + context_size - 2], smiles[i + context_size - 1], smiles[i + context_size]]  # after el
#             target = smiles[i]

#             # Tuple together context and target
#             single_smiles_data.append((context, target))

#         # Append all the data together but preserve the separation of context between smiles
#         context_val += single_smiles_data

#     # Epoch loss history for plotting
#     train_loss_history = []
#     val_loss_history = []

#     # Loss function, model, and optimizer
#     criterion = nn.NLLLoss()
#     model = CBOW(vocab=vocab, embedding_dim=embedding_size, context=2*context_size).to(device)
#     optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#     for epoch in range(epochs):
#         print('EPOCH', epoch)

#         # Initialize total loss over one epoch
#         epoch_train_loss = []
#         epoch_val_loss = []

#         # Run through TRAIN batches
#         for context, target in context_train:
#             # Zero out gradient every batch
#             model.zero_grad()

# #             # Cast context and target to tensor
# #             context = torch.tensor(context, dtype=torch.long).to(device)
# #             target = torch.tensor([target], dtype=torch.long).to(device)

# #             # Make predictions
# #             log_probs = model(context)

# #             # Calculate loss
# #             loss = criterion(log_probs, target)

# #             # Take train step
# #             loss.backward()
# #             optimizer.step()

# #             # Compile loss
# #             epoch_train_loss.append(loss.item())
# #         train_loss_history.append(sum(epoch_train_loss) / len(epoch_train_loss))

# #         # Run through VAL batches
# #         for context, target in context_val:
# #             # Cast context and target to tensor
# #             context = torch.tensor(context, dtype=torch.long).to(device)
# #             target = torch.tensor([target], dtype=torch.long).to(device)

# #             # Set model to evaluation mode
# #             model.eval()
# #             with torch.no_grad():

# #                 # Make predictions
# #                 log_probs = model(context)
# #             model.train()

# #             # Calculate loss
# #             loss = criterion(log_probs, target)

# #             # Compile loss
# #             epoch_val_loss.append(loss.item())
# #         val_loss_history.append(sum(epoch_val_loss) / len(epoch_val_loss))

# #     # Plot loss over epochs
# #     plt.figure()
# #     plt.title('Negative log likelihood loss')
# #     plt.plot(range(epochs), train_loss_history, label='Train loss')
# #     plt.plot(range(epochs), val_loss_history, label='Validation loss')
# #     plt.xlabel('Epochs')
# #     plt.ylabel('Loss')
# #     plt.legend()
# #     plt.savefig('word_embedding_loss_plot')
# #     plt.show()

# #     torch.save(model.state_dict(), 'word_embedding_trained_model.pth')


# if __name__ == '__main__':
#     # Enable GPU if available
#     device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#     print(device)

#     # Read in data
#     data = pd.read_csv('word_embedding_train_sample.csv', header=None)[1]
#     print(data, data.shape)

#     # Declarations
#     context_size = 3
#     embedding_size = 16
#     learning_rate = 0.001
#     epochs = 20

#     # Split data
#     train_data, val_data = train_test_split(data, test_size=0.2, random_state=0)

#     # Load pre-generated vocabulary
#     vocab = pd.read_csv('anagenex_chembl_vocab.txt', header=None).to_numpy()
#     vocab = np.insert(vocab, 0, '<unk>')
#     vocab = np.insert(vocab, 0, '<pad>')  # insert padding entry to vocab

#     # Label encode SMILES by index in vocab
#     encoded_train = encode_smiles(train_data, vocabulary=vocab)
#     encoded_val = encode_smiles(val_data, vocabulary=vocab)

#     # Run train and validation loop
#     training_loop()



# input text
# bert
# probability output for each word
# argmax

# patient has chest pain
# vector length features [None]