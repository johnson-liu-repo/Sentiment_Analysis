
### This code is not used. Saved for future reference.

# TF-IDF 
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import numpy as np




def compute_tf_idf():
    pass



# ---------------------------
# TF-IDF Weighted Comment Vector
# ---------------------------
def embed_comment_tfidf(comment, vectorizer, word_to_idx, embedding_matrix):
    """
    Convert a comment to a single vector using TF-IDF weighted word embeddings.
    """
    tfidf = vectorizer.transform([comment])
    feature_names = vectorizer.get_feature_names_out()

    vectors = []
    weights = []

    # Loop over non-zero TF-IDF features in the comment
    for col in tfidf.nonzero()[1]:
        word = feature_names[col]
        weight = tfidf[0, col]
        idx = word_to_idx.get(word)

        # If word is in vocab, apply its TF-IDF weight to the embedding
        if idx is not None:
            vectors.append(embedding_matrix[idx])
            weights.append(weight)

    if vectors:
        vectors = torch.stack(vectors)
        weights = torch.tensor(weights).unsqueeze(1)
        weighted = vectors * weights
        return weighted.sum(dim=0) / weights.sum()  # Normalize by total weight
    else:
        return torch.zeros(embedding_matrix.shape[1])  # fallback if no known words


if __name__ == "__main__":
    # Example: Embed a sarcastic comment


    # ---------------------------
    # Sample text corpus: Reddit-like comments
    # ---------------------------
    comments = [
        "this is great",
        "oh sure that will work",
        "what a fantastic plan",
        "yeah totally not sarcasm",
        "this plan will totally work",
        "not sure that is a great plan",
        "yeah right, fantastic idea",
        "sure this is sarcasm"
    ]

    # ---------------------------
    # Step 1: Fit a TF-IDF Vectorizer
    # ---------------------------
    # This calculates both TF and IDF values across the corpus
    vectorizer = TfidfVectorizer(lowercase=True, tokenizer=str.split, token_pattern=None)
    vectorizer.fit(comments)

    # ---------------------------
    # Step 2: Build Vocabulary and Fake Word Embeddings
    # ---------------------------
    # Vocabulary: word → index
    word_to_idx = {word: idx for idx, word in enumerate(vectorizer.get_feature_names_out())}

    # Fake embeddings for demonstration (dim=50)
    embedding_dim = 50
    embedding_matrix = torch.randn(len(word_to_idx), embedding_dim)
    print(embedding_matrix)

    # ---------------------------
    # Step 3: TF-IDF Weighted Comment Vector
    # ---------------------------
    comment_vector = embed_comment_tfidf("oh sure that will work", vectorizer, word_to_idx, embedding_matrix)

    print("Comment vector shape:", comment_vector.shape)
    print("Comment vector:", comment_vector)