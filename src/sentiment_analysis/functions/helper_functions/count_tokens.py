
import pandas as pd
from collections import Counter

def count_total_tokens(sentences, lowercase=True):
    """
    Counts total number of tokens (words) in a list of text strings.

    Args:
        sentences (list of str): List of comments or sentences.
        lowercase (bool): Whether to lowercase before tokenizing.

    Returns:
        total_tokens (int): Total number of word tokens.
        vocab_size (int): Number of unique words.
        word_freqs (Counter): Frequency of each word.
    """
    word_freqs = Counter()
    total_tokens = 0

    for sentence in sentences:
        if lowercase:
            sentence = sentence.lower()
        words = sentence.split()
        word_freqs.update(words)
        total_tokens += len(words)

    vocab_size = len(word_freqs)
    return total_tokens, vocab_size, word_freqs


if __name__ == '__main__':
    data_file = 'data/project_data/raw_data/trimmed_training_data.csv'

    df = pd.read_csv(data_file)
    comments_list = df['comment'].dropna().astype(str).tolist()

    total_tokens, vocab_size, word_freqs = count_total_tokens(comments_list)

    print(f"Total tokens: {total_tokens:,}")
    print(f"Vocabulary size: {vocab_size:,}")
    print("Top 10 most frequent words:", word_freqs.most_common(10))