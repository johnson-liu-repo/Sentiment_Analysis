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


def check_cosine_similarity(word_list, embedding_matrix, word_to_idx, plot=True):
    vectors = []
    for word in word_list:
        idx = word_to_idx.get(word)
        if idx is None:
            raise ValueError(f"Word '{word}' not in vocabulary.")
        vectors.append(embedding_matrix[idx])
    
    vectors = torch.stack(vectors)
    sim_matrix = F.cosine_similarity(vectors.unsqueeze(1), vectors.unsqueeze(0), dim=-1).numpy()
    
    if plot:
        plt.figure(figsize=(8, 6))
        sns.heatmap(sim_matrix, xticklabels=word_list, yticklabels=word_list, annot=True, fmt=".2f", cmap="viridis")
        plt.title("Cosine Similarity Between Words")
        plt.show()
    
    return sim_matrix


if __name__ == '__main__':
    trained_word_vectors_file = 'testing_scrap_misc/training_01/word_vector_training/training_logs/weights_epoch_57.pt'
    word_vectors_matrix = torch.load(trained_word_vectors_file)
    
    unique_words_file = 'testing_scrap_misc/training_01/preprocessing/unique_words.npy'
    unique_words = np.load(unique_words_file)

    word_to_idx = generate_word_to_idx(unique_words)

    word_list = ['dog', 'cat', 'king', 'queen', 'man', 'woman']

    sim_matrix = check_cosine_similarity(word_list, word_vectors_matrix, word_to_idx)