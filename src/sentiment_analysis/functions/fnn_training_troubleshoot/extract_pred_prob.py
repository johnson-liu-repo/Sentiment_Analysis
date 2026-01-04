import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.nn as nn

# -----------------------
# Load original data
# -----------------------
# Replace with your actual data
X = np.load('testing_scrap_misc/training_01/fnn/vectorized_comments.npy')
y = np.load('testing_scrap_misc/training_01/preprocessing/labels.npy')

# Convert to tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# -----------------------
# Create validation set
# -----------------------
val_ratio = 0.2  # use 20% as validation
val_size = int(len(X_tensor) * val_ratio)
train_size = len(X_tensor) - val_size
_, val_set = random_split(TensorDataset(X_tensor, y_tensor), [train_size, val_size])

val_loader = DataLoader(val_set, batch_size=32)

# -----------------------
# Model definition (must match training)
# -----------------------
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

            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

# -----------------------
# Load model
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FeedforwardNN(X.shape[1])
model.load_state_dict(torch.load('testing_scrap_misc/training_01/fnn/best_fnn_model.pt', map_location=device))
model.to(device)
model.eval()

# -----------------------
# Run predictions
# -----------------------
all_probs = []
all_preds = []
all_targets = []

with torch.no_grad():
    for X_batch, y_batch in val_loader:
        X_batch = X_batch.to(device)
        logits = model(X_batch)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        preds = (probs > 0.5).astype(int)
        targets = y_batch.numpy().flatten()

        all_probs.extend(probs)
        all_preds.extend(preds)
        all_targets.extend(targets)

# -----------------------
# Save outputs
# -----------------------
np.save("val_probs.npy", np.array(all_probs))
np.save("val_preds.npy", np.array(all_preds))
np.save("val_targets.npy", np.array(all_targets))
