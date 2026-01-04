def visualize_cosine_similarity(vectors):
    """
    Visualizes the cosine similarity between vectors using a heatmap.

    Args:
        1. vectors (list of list of float): List of vectors to visualize.
    
    The Function:
        1. Computes the cosine similarity between the vectors.
        2. Creates a heatmap to visualize the cosine similarity matrix.
    
    Returns:
        To terminal:
            1. None
        To file:
            1. None
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Calculate cosine similarity matrix
    from sklearn.metrics.pairwise import cosine_similarity
    similarity_matrix = cosine_similarity(vectors)

    # Ignore self-similarity by setting the diagonal to NaN
    np.fill_diagonal(similarity_matrix, np.nan)

    # Create a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, annot=True, cmap='coolwarm', fmt=".2f",
                xticklabels=[f'Vector {i+1}' for i in range(len(vectors))],
                yticklabels=[f'Vector {i+1}' for i in range(len(vectors))])
    plt.title('Cosine Similarity Heatmap (Ignoring Self-Similarity)')
    plt.show()


### This doesn't work as expected ... might look back into this in the future.
# def visualize_cosine_similarity_2(vectors):
#     # Visualize the cosine similarity between vectors using a heatmap while sorting the map by the overall similarity.
#     import numpy as np
#     import matplotlib.pyplot as plt
#     import seaborn as sns
#     from sklearn.metrics.pairwise import cosine_similarity
    
#     # Calculate cosine similarity matrix
#     similarity_matrix = cosine_similarity(vectors)
    
#     # Sort the similarity matrix and tick labels based on the sum of similarities
#     sorted_indices = np.argsort(-np.sum(similarity_matrix, axis=1))  # Sort by descending row sums
#     sorted_similarity_matrix = similarity_matrix[sorted_indices][:, sorted_indices]
#     sorted_tick_labels = [f'Vector {i+1}' for i in sorted_indices]

#     # Create the heatmap
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(sorted_similarity_matrix, annot=True, cmap='coolwarm', fmt=".2f",
#                 xticklabels=sorted_tick_labels,  # Use the sorted tick labels
#                 yticklabels=sorted_tick_labels)
#     plt.title('Cosine Similarity Heatmap (Sorted by Overall Similarity)')
#     plt.show()


if __name__ == "__main__":
    import numpy as np

    file_name = 'word_vectors_over_time_02.npy'

    trained_vectors = np.load(file_name, allow_pickle=True)[-1]

    vectors = [trained_vectors[word] for word in trained_vectors]

    visualize_cosine_similarity(vectors)