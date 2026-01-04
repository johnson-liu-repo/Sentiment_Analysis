import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.metrics import f1_score
from tqdm import tqdm




def custom_cnn( X,
                labels: np.ndarray,
                epochs: int = 10,
                patience: int = 10,
                batch_size: int = 32,
                learning_rate: float = 0.01 ):
    ############################################################################
    # Model
    class ConvolutionalNN(nn.Module):
        def __init__(self, embedding_dim, num_classes):
            super(ConvolutionalNN, self).__init__()
            self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=100, kernel_size=3)
            self.relu = nn.ReLU()
            self.pool = nn.AdaptiveMaxPool1d(1)
            self.fc = nn.Linear(100, num_classes)
            self.dropout = nn.Dropout(0.5)

        def forward(self, x):  # x: [batch_size, sequence_length, embedding_dim]
            x = x.permute(0, 2, 1)  # -> [batch_size, embedding_dim, sequence_length]
            x = self.conv1(x)       # -> [batch_size, out_channels, L_out]
            x = self.relu(x)
            x = self.pool(x).squeeze(2)  # -> [batch_size, out_channels]
            x = self.dropout(x)
            x = self.fc(x)          # -> [batch_size, num_classes]
            return x

    ############################################################################
    # Hyperparameters
    num_classes = 2

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    embedding_dim = X_train.shape[2]
    model = ConvolutionalNN(embedding_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), learning_rate)

    # Dataloaders
    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=batch_size)

    # Training loop
    for epoch in range(epochs):
        model.train()
        all_preds, all_labels = [], []
        total_loss = 0

        for X_batch, Y_batch in tqdm(train_loader):
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(Y_batch.cpu().numpy())

        f1 = f1_score(all_labels, all_preds, average='macro')
        print(f"Epoch {epoch+1} | Train Loss: {total_loss:.4f} | Train F1: {f1:.4f}")

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        val_loss = 0
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                outputs = model(X_batch)
                val_loss += criterion(outputs, Y_batch).item()
                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(Y_batch.cpu().numpy())

        val_f1 = f1_score(val_labels, val_preds, average='macro')
        print(f"           | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")
