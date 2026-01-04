import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import glob
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm


def _latest_checkpoint_path(ckpt_dir: str) -> str | None:
    pattern = os.path.join(ckpt_dir, "fnn_epoch_*.pt")
    paths = glob.glob(pattern)
    if not paths:
        # Fallback to last_checkpoint.pt if present
        last_ckpt = os.path.join(ckpt_dir, "last_checkpoint.pt")
        return last_ckpt if os.path.exists(last_ckpt) else None
    # Choose the newest by epoch number if possible; else by mtime
    def _epoch_num(p):
        import re
        m = re.search(r"fnn_epoch_(\d+).pt$", os.path.basename(p))
        return int(m.group(1)) if m else -1
    paths.sort(key=lambda p: (_epoch_num(p), os.path.getmtime(p)))
    return paths[-1]


def _save_checkpoint(path: str, *, model, optimizer, epoch: int, best_val_f1: float, wait: int) -> None:
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val_f1": best_val_f1,
        "wait": wait,
    }
    torch.save(payload, path)


def _load_checkpoint(path: str, *, model, optimizer):
    data = torch.load(path, map_location="cpu")
    model.load_state_dict(data["model_state"])
    optimizer.load_state_dict(data["optimizer_state"])
    epoch = int(data.get("epoch", -1))
    best_val_f1 = float(data.get("best_val_f1", 0.0))
    wait = int(data.get("wait", 0))
    return epoch, best_val_f1, wait


def custom_fnn( X: np.ndarray,
                labels: np.ndarray,
                epochs: int = 10,
                patience: int = 10,
                batch_size: int = 32,
                learning_rate: float = 0.001,
                num_workers: int = 0,
                seed: int = 94,
                # NEW:
                checkpoint_every: int = 1,
                resume: bool = False,
                resume_path: str | None = None ):
    """
    Train a feedforward network on pre-vectorized features.

    New arguments:
      - checkpoint_every: save a training checkpoint every N epochs (and always on final epoch).
      - resume: if True, resume from the most recent checkpoint in model_checkpoints/ (or from resume_path).
      - resume_path: explicit path to a checkpoint .pt file to resume from. Overrides auto-discovery.
    """
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
    class FeedforwardNN(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
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

    # Scale before converting to tensors
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Torch tensors
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(labels, dtype=torch.float32)

    # Deterministic split (seeded by torch.manual_seed above)
    dataset = TensorDataset(X_tensor, y_tensor)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    # Dataloaders
    pin = (device.type == "cuda")
    train_loader = DataLoader(train_set, batch_size, num_workers=num_workers, shuffle=True, pin_memory=pin)
    val_loader   = DataLoader(val_set,   batch_size, num_workers=num_workers, pin_memory=pin)

    # Model / optimizer / loss
    model = FeedforwardNN(X_tensor.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Book-keeping
    best_val_f1 = 0.0
    wait = 0
    start_epoch = 0

    model_save_dir = 'data/training_data/test_training_02/fnn/'
    ckpt_dir = os.path.join(model_save_dir, 'model_checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    # Initialize/append log
    log_path = os.path.join(model_save_dir, 'fnn_training_log.csv')
    log_columns = ["epoch", "train_loss", "val_loss", "val_acc", "val_f1_macro", "val_f1_micro"]
    if os.path.exists(log_path):
        try:
            log_df = pd.read_csv(log_path)
        except Exception:
            log_df = pd.DataFrame(columns=log_columns)
    else:
        log_df = pd.DataFrame(columns=log_columns)

    # ---- RESUME ----
    if resume:
        ckpt_to_load = resume_path if resume_path else _latest_checkpoint_path(ckpt_dir)
        if ckpt_to_load and os.path.exists(ckpt_to_load):
            print(f"Resuming from checkpoint: {ckpt_to_load}")
            last_epoch, best_val_f1, wait = _load_checkpoint(ckpt_to_load, model=model, optimizer=optimizer)
            start_epoch = (last_epoch + 1)
            print(f"Start epoch set to {start_epoch} (best_val_f1 so far: {best_val_f1:.4f}, early-stop wait: {wait})")
        else:
            print("Resume requested but no checkpoint found; starting from scratch.")

    # ---- TRAIN LOOP ----
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Training]", leave=False)
        for X_batch, y_batch in train_bar:
            X_batch = X_batch.to(device, non_blocking=pin)
            y_batch = y_batch.to(device, non_blocking=pin).unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Validation]", leave=False)
        with torch.no_grad():
            for X_val, y_val in val_bar:
                X_val = X_val.to(device, non_blocking=pin)
                y_val = y_val.to(device, non_blocking=pin).unsqueeze(1)

                logits = model(X_val)
                loss = criterion(logits, y_val)
                val_loss += loss.item()

                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(y_val.cpu().numpy())

        val_preds = np.array(val_preds).flatten()
        val_targets = np.array(val_targets).flatten()
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1_macro = f1_score(val_targets, val_preds, average="macro")
        val_f1_micro = f1_score(val_targets, val_preds, average="micro")

        avg_train_loss = total_loss / max(1, len(train_loader))
        avg_val_loss = val_loss / max(1, len(val_loader))

        # Console summary
        print(f"Epoch {epoch + 1}:"
              f" Train Loss={avg_train_loss:.3f} |"
              f" Val Loss={avg_val_loss:.3f} |"
              f" Val F1 Macro={val_f1_macro:.3f} |"
              f" Val Acc={val_acc:.3f}")

        # Logging
        log_row = pd.DataFrame([[epoch + 1, avg_train_loss, avg_val_loss, val_acc, val_f1_macro, val_f1_micro]],
                               columns=log_columns)
        log_df = pd.concat([log_df, log_row], ignore_index=True)
        log_df.to_csv(log_path, index=False)

        # Early stopping on macro-F1
        improved = val_f1_macro > best_val_f1
        if improved:
            best_val_f1 = val_f1_macro
            wait = 0
            torch.save(model.state_dict(), os.path.join(model_save_dir, 'best_fnn_model.pt'))
        else:
            wait += 1

        # ---- CHECKPOINT SAVE ----
        need_ckpt = (checkpoint_every is not None and checkpoint_every > 0 and ((epoch + 1) % checkpoint_every == 0)) or ((epoch + 1) == epochs)
        if need_ckpt:
            epoch_ckpt = os.path.join(ckpt_dir, f"fnn_epoch_{epoch+1:03d}.pt")
            _save_checkpoint(epoch_ckpt, model=model, optimizer=optimizer, epoch=epoch, best_val_f1=best_val_f1, wait=wait)
            # Also update "last_checkpoint.pt" for convenience
            torch.save(torch.load(epoch_ckpt, map_location="cpu"), os.path.join(ckpt_dir, "last_checkpoint.pt"))

        # Early stop check
        if wait >= patience:
            print("Early stopping.")
            break

    print(f"Best validation F1: {best_val_f1:.4f}")