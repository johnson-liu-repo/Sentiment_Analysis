


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import numpy as np
import os
import pandas as pd
import random

def custom_fnn( X: np.ndarray,
                labels: np.ndarray,
                epochs: int = 10,
                patience: int = 10,
                batch_size: int = 32,
                learning_rate: float = 0.001,
                num_workers: int = 0,
                seed: int = 94 ):
    ############################################################################
    # Set random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    ############################################################################
    # Model
    ### Note:
    #       - Later on, when we extend the project to more
    #         classifications (sentiment, tone, etc.), we
    #         need to specify the num_classes for the
    #         expected dimension of the output of the NN.
    class FeedforwardNN(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            # self.net = nn.Sequential(
            #     nn.Linear(input_dim, 32),
            #     nn.BatchNorm1d(32),
            #     nn.ReLU(),
            #     nn.Dropout(0.3),

            #     nn.Linear(32, 32),
            #     nn.BatchNorm1d(32),
            #     nn.ReLU(),
            #     nn.Dropout(0.3),

            #     nn.Linear(32, 1)  # output logit for binary classification
            # )
            self.net = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),

                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3),

                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.3),

                nn.Linear(64, 1)  # Single logit output
            )
        def forward(self, x):
            return self.net(x)

    ############################################################################

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Convert to tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    X = StandardScaler().fit_transform(X)

    y_tensor = torch.tensor(labels, dtype=torch.float32)

    # Train/Val split
    dataset = TensorDataset(X_tensor, y_tensor)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size, num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(val_set, batch_size, num_workers=num_workers)

    model = FeedforwardNN(X.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    # optimizer = optim.Adam(model.parameters(), learning_rate)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)


    best_val_f1 = 0
    wait = 0

    train_losses, val_accuracies = [], []
    val_f1_macros, val_f1_micros = [], []

    model_save_dir = 'testing_scrap_misc/training_01/fnn/'
    os.makedirs(model_save_dir + 'model_checkpoints', exist_ok=True)

    # Initialize log dataframe
    log_path = model_save_dir + 'fnn_training_log.csv'
    log_columns = ["epoch", "train_loss", "val_loss", "val_acc", "val_f1_macro", "val_f1_micro"]
    log_df = pd.DataFrame(columns=log_columns)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Training]", leave=False)
        for X_batch, y_batch in train_bar:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(1)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())

        model.eval()

        val_loss = 0
        val_preds, val_targets = [], []
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Validation]", leave=False)
        with torch.no_grad():
            for X_val, y_val in val_bar:
                X_val = X_val.to(device)
                y_val = y_val.to(device).unsqueeze(1)

                logits = model(X_val)
                loss = criterion(logits, y_val)
                val_loss += loss.item()

                probs = torch.sigmoid(logits)  # Use logits from above
                preds = (probs > 0.5).float()
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(y_val.cpu().numpy())

        val_preds = np.array(val_preds).flatten()
        val_targets = np.array(val_targets).flatten()
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1_macro = f1_score(val_targets, val_preds, average="macro")
        val_f1_micro = f1_score(val_targets, val_preds, average="micro")
    
        avg_train_loss = total_loss / len(train_loader)  # <-- Average loss
        avg_val_loss = val_loss / len(val_loader)        # <-- Average val loss

        train_losses.append(avg_train_loss)
        val_accuracies.append(val_acc)
        val_f1_macros.append(val_f1_macro)
        val_f1_micros.append(val_f1_micro)

        print(f"Epoch {epoch + 1}:\n"
              f"Train Loss={avg_train_loss:.3f} | "
              f"Val Loss={avg_val_loss:.3f} | "
              f"Val F1 Macro={val_f1_macro:.3f} | "
              f"Val Acc={val_acc:.3f} "
              f"\n")

        # Save model for this epoch (preferably state_dict)
        # torch.save(model.state_dict(), model_save_dir + f'model_checkpoints/fnn_epoch_{epoch+1:02d}.pt')

        log_row = pd.DataFrame([[epoch + 1, avg_train_loss, avg_val_loss, val_acc, val_f1_macro, val_f1_micro]], columns=log_columns)
        log_df = pd.concat([log_df, log_row], ignore_index=True)
        log_df.to_csv(log_path, index=False)

        # Use macro F1 for early stopping (or micro, as you prefer)
        if val_f1_macro > best_val_f1:
            best_val_f1 = val_f1_macro
            wait = 0
            torch.save(model.state_dict(), model_save_dir + 'best_fnn_model.pt')
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping.")
                break

    print(f"Best validation F1: {best_val_f1:.4f}")
