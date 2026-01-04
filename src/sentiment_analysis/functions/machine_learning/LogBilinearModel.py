

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import glob
from tqdm import tqdm



# class SparseCoocDataset(Dataset):
#     def __init__(self, i, j, log_count, weight):
#         self.i = i
#         self.j = j
#         self.log_count = log_count
#         self.weight = weight



class SparseCoocDataset(Dataset):
    def __init__(self, i, j, log_count, weight):
        self.i = i
        self.j = j
        self.log_count = log_count
        self.weight = weight

    def __len__(self):
        return len(self.i)

    def __getitem__(self, idx):
        return self.i[idx], self.j[idx], self.log_count[idx], self.weight[idx]

class LogBilinearModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.word_biases = nn.Embedding(vocab_size, 1)
        self.context_biases = nn.Embedding(vocab_size, 1)

    def forward(self, word_idx, context_idx):
        w = self.word_embeddings(word_idx)
        c = self.context_embeddings(context_idx)
        b_w = self.word_biases(word_idx).squeeze()
        b_c = self.context_biases(context_idx).squeeze()
        dot = torch.sum(w * c, dim=1)
        return dot + b_w + b_c

def weighting_function(x, x_max=100, alpha=0.75):
    wx = (x / x_max) ** alpha
    return torch.where(x < x_max, wx, torch.ones_like(x))

def find_latest_checkpoint(directory):
    checkpoints = glob.glob(os.path.join(directory, 'checkpoint_epoch_*.pt'))
    if not checkpoints:
        return None
    latest = max(checkpoints, key=lambda f: int(f.split('_')[-1].split('.')[0]))
    return latest

def train_sparse_glove(
        cooc_sparse,
        embedding_dim=200,
        epochs=100,
        batch_size=256,
        learning_rate=0.001,
        x_max=100,
        alpha=0.75,
        num_workers=4,
        training_save_dir='training_logs/',
        use_gpu=True,
        resume_checkpoint=False,
        checkpoint_interval=1
    ):
    checkpoint_dir = training_save_dir + 'training_logs/'

    os.makedirs(checkpoint_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")

    i = torch.tensor(cooc_sparse.row, dtype=torch.long)
    j = torch.tensor(cooc_sparse.col, dtype=torch.long)
    x_ij = torch.tensor(cooc_sparse.data, dtype=torch.float32)
    log_count = torch.log(x_ij + 1e-10)
    weight = weighting_function(x_ij, x_max=x_max, alpha=alpha)

    # -------------------------------------------
    # dataset = SparseCoocDataset(i, j, x_ij)
    # -------------------------------------------
    dataset = SparseCoocDataset(i, j, log_count, weight)
    val_size = int(len(dataset) * 0.1)
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(94)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    vocab_size = cooc_sparse.shape[0]

    model = LogBilinearModel(vocab_size, embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    start_epoch = 0

    if resume_checkpoint:
        latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            checkpoint = torch.load(latest_checkpoint)
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resumed training from {latest_checkpoint}")

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0
        for word_idx, context_idx, log_count_batch, weight_batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            
            word_idx = word_idx.to(device)
            context_idx = context_idx.to(device)
            log_count_batch = log_count_batch.to(device)
            weight_batch = weight_batch.to(device)
            log_count_batch = log_count_batch.to(device)
            weight_batch = weight_batch.to(device)

            optimizer.zero_grad()
            prediction = model(word_idx, context_idx)
            # --------------------------------------------------------------
            # log_count = torch.log(count + 1e-10)
            # weight = weighting_function(count, x_max=x_max, alpha=alpha)
            # loss = weight * (prediction - log_count) ** 2
            # --------------------------------------------------------------
            loss = weight_batch.to(device) * (prediction - log_count_batch.to(device)) ** 2
            loss = loss.mean()
            loss.backward()            

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        log_line = f"Epoch {epoch + 1}, Loss: {avg_loss:.6f}\n"
        print(log_line.strip())

        with open(os.path.join(training_save_dir, "training_log.txt"), "a") as f:
            f.write(log_line)

        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        if new_lr < old_lr:
            print(f"Learning rate reduced: {old_lr:.6f} --> {new_lr:.6f}")
            with open(os.path.join(training_save_dir, "training_log.txt"), "a") as f:
                f.write(f"Learning rate reduced: {old_lr:.6f} --> {new_lr:.6f}\n")

        if (epoch + 1) % checkpoint_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
            }, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pt'))

    torch.save(model.word_embeddings.weight.cpu().detach(), os.path.join(training_save_dir, 'final_word_vectors.pt'))
