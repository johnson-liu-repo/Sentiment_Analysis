
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def generate_word_to_idx(vocab, add_special_tokens=False):
    """
    Generate a word-to-index dictionary from a list of unique words.

    Args:
        vocab (List[str]): List of unique words.
        add_special_tokens (bool): Whether to add <PAD> and <UNK> tokens at the start.

    Returns:
        Dict[str, int]: Mapping from word to integer index.
    """
    word_to_idx = {}

    idx = 0

    if add_special_tokens:
        word_to_idx['<PAD>'] = idx
        idx += 1
        word_to_idx['<UNK>'] = idx
        idx += 1

    for word in vocab:
        if word not in word_to_idx:  # skip if already in (e.g. duplicates)
            word_to_idx[word] = idx
            idx += 1
    
    return word_to_idx

def verify_word_embedding_alignment(word_to_idx, embedding_matrix, sample_words=10):
    """
    Verifies that the word_to_idx mapping aligns with the embedding matrix.
    Picks a sample of words, retrieves their indices, and checks vector differences.

    Args:
        word_to_idx (dict): word → index mapping
        embedding_matrix (Tensor or nn.Embedding): shape [vocab_size, embedding_dim]
        sample_words (int): Number of words to sample for spot-checking

    Returns:
        List of mismatches (if any)
    """
    import random
    import torch

    # Convert to tensor if nn.Embedding
    if hasattr(embedding_matrix, 'weight'):
        embedding_tensor = embedding_matrix.weight.data
    else:
        embedding_tensor = embedding_matrix

    vocab_size = embedding_tensor.shape[0]

    # Rebuild idx_to_word
    idx_to_word = {i: w for w, i in word_to_idx.items()}
    
    # Check for out-of-range issues
    if max(word_to_idx.values()) >= vocab_size:
        print("❌ Error: Some word indices exceed embedding_matrix size.")
        return

    # Sample some word:index pairs
    sampled = random.sample(list(word_to_idx.items()), min(sample_words, len(word_to_idx)))
    mismatches = []

    for word, idx in sampled:
        recovered_idx = None
        recovered_word = idx_to_word.get(idx, None)
        if recovered_word != word:
            mismatches.append((word, idx, recovered_word))

    if not mismatches:
        print("✅ `word_to_idx` appears to be correctly aligned with `embedding_matrix`.")
    else:
        print("❌ Misalignment found!")
        for word, idx, recovered in mismatches:
            print(f"Expected word '{word}' at index {idx}, but idx_to_word[{idx}] = '{recovered}'")

    return mismatches


if __name__ == '__main__':
    
    trained_word_vectors_file = 'testing_scrap_misc/training_01/word_vector_training/training_logs/weights_epoch_57.pt'
    word_vectors_matrix = torch.load(trained_word_vectors_file)
    
    unique_words_file = 'testing_scrap_misc/training_01/preprocessing/unique_words.npy'
    unique_words = np.load(unique_words_file)

    word_to_idx = generate_word_to_idx(unique_words)

    verify_word_embedding_alignment(word_to_idx, word_vectors_matrix)