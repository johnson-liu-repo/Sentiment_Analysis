

'''
import subprocess
preprocess = ['python3', 'main.py', '--part', 'preprocess_data']
train_word_vectors = ['python3', 'main.py', '--part', 'train_word_vectors']
train_fnn = ['python3', 'main.py', '--part', 'train_fnn']
subprocess.run(preprocess)
'''




top_n_words = 10

import seaborn as sns
# Load the co-occurrence matrix
from scipy.sparse import load_npz
M = load_npz('data/training_data/test_training_02/preprocessing/cooccurrence_matrix.npz')

# Load the unique words
import numpy as np
unique_words = np.load('data/training_data/test_training_02/preprocessing/unique_words.npy', allow_pickle=True)

# Create a mapping from words to indices
word_to_idx = {word: idx for idx, word in enumerate(unique_words)}
top_idxs = [word_to_idx[w] for w in unique_words[:top_n_words]]

# Convert M to 2D array.
import matplotlib.pyplot as plt
import scipy.sparse as sp
if sp.issparse(M):
    M = M.todense()  # Convert sparse matrix to dense if needed

sub = M[top_idxs, :][:, top_idxs]
sns.heatmap(
    sub,
    xticklabels=unique_words[:top_n_words],
    yticklabels=unique_words[:top_n_words],
    cmap='viridis',
    annot=True,
)

plt.xticks(rotation=90)
plt.show()
