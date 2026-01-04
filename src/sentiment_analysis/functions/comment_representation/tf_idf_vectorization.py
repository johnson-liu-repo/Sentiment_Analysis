import numpy as np
import torch

def vectorize_comments_with_tfidf(text, vectorizer, word_vectors_matrix, output_file_name):
    """
    Vectorizes each comment in `text` using TF-IDF weighted word embeddings and saves the result to `output_file`.

    Args:
        text (array-like): List or array of comment strings.
        vectorizer (TfidfVectorizer): Fitted sklearn TfidfVectorizer.
        word_vectors_matrix (np.ndarray or torch.Tensor): Matrix of word embeddings (vocab_size x embedding_dim).
        output_file_name (str): Path to save the resulting numpy array.
    """
    # Get the TF-IDF matrix for all comments (num_comments x vocab_size)
    tfidf_matrix = vectorizer.transform(text)  # Usually a sparse matrix

    # Ensure word_vectors_matrix is a numpy array (vocab_size x embedding_dim)
    if isinstance(word_vectors_matrix, torch.Tensor):
        word_vectors_matrix = word_vectors_matrix.cpu().numpy()

    # print('tfidf_matrix:')
    # print(tfidf_matrix[:3])
    # print('word_vectors_matrix:')
    # print(word_vectors_matrix[:3])


    # Multiply: (num_comments x vocab_size) @ (vocab_size x embedding_dim) = (num_comments x embedding_dim)
    vectorized_comments = tfidf_matrix @ word_vectors_matrix

    # Convert to dense if needed
    if not isinstance(vectorized_comments, np.ndarray):
        vectorized_comments = vectorized_comments.toarray()

    np.save(output_file_name, vectorized_comments)
    print(f'Vectorized comments saved to {output_file_name}.')