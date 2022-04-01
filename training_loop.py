import sys
import torch

sys.path.append('../ml_tools/src/ml_tools')
import torch_utils
from model_architecture import RNNDataset, RNNModel


# Set device
device = torch_utils.get_device()


def training_loop():
    # Model params
    batch_size = 16
    hidden_size = 64
    num_features = X.shape[0]
    rnn_layers = 1

    # Epoch loss history for plotting
    train_loss_history = []
    val_loss_history = []

    # Loss function, model, and optimizer
    criterion = nn.NLLLoss()
    model = CBOW(vocab=vocab, embedding_dim=embedding_size, context=2*context_size).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        print('EPOCH', epoch)

        # Initialize total loss over one epoch
        epoch_train_loss = []
        epoch_val_loss = []

        # Run through TRAIN batches
        for context, target in context_train:
            # Zero out gradient every batch
            model.zero_grad()

            # Cast context and target to tensor
            context = torch.tensor(context, dtype=torch.long).to(device)
            target = torch.tensor([target], dtype=torch.long).to(device)

            # Make predictions
            log_probs = model(context)

            # Calculate loss
            loss = criterion(log_probs, target)

            # Take train step
            loss.backward()
            optimizer.step()

            # Compile loss
            epoch_train_loss.append(loss.item())
        train_loss_history.append(sum(epoch_train_loss) / len(epoch_train_loss))

        # Run through VAL batches
        for context, target in context_val:
            # Cast context and target to tensor
            context = torch.tensor(context, dtype=torch.long).to(device)
            target = torch.tensor([target], dtype=torch.long).to(device)

            # Set model to evaluation mode
            model.eval()
            with torch.no_grad():

                # Make predictions
                log_probs = model(context)
            model.train()

            # Calculate loss
            loss = criterion(log_probs, target)

            # Compile loss
            epoch_val_loss.append(loss.item())
        val_loss_history.append(sum(epoch_val_loss) / len(epoch_val_loss))

    # Plot loss over epochs
    plt.figure()
    plt.title('Negative log likelihood loss')
    plt.plot(range(epochs), train_loss_history, label='Train loss')
    plt.plot(range(epochs), val_loss_history, label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('word_embedding_loss_plot')
    plt.show()

    torch.save(model.state_dict(), 'word_embedding_trained_model.pth')