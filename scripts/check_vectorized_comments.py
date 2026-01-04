import numpy as np

def print_comment_vectors(start_idx=0, end_idx=5):
    """
    Load and print vectorized comments from a specific index range.
    
    Args:
        start_idx (int): Starting index (inclusive)
        end_idx (int): Ending index (exclusive) 
    """
    # Load the vectorized comments
    vectors_path = 'data/training_data/test_training_02/pretrained/vectorized_comments.npy'
    vectors = np.load(vectors_path, allow_pickle=True)
    
    # Basic validation
    if start_idx < 0 or end_idx > len(vectors):
        raise ValueError(f"Index range [{start_idx}:{end_idx}] out of bounds. Array length: {len(vectors)}")
    
    # Print info about the vectors
    print(f"Total number of vectors: {len(vectors)}")
    print(f"Vector dimension: {vectors[0].shape}")
    print("\nVectors from index {start_idx} to {end_idx-1}:")
    
    # Print the requested range of vectors
    for idx in range(start_idx, end_idx):
        print(f"\nVector {idx}:")
        print(vectors[idx])

# Example usage
if __name__ == "__main__":
    # print_comment_vectors(0, 3)  # Print first 3 vectors
    # print len(vectors)  # Print total number of vectors
    print(f"Total number of vectors: {len(np.load('data/training_data/test_training_02/pretrained/vectorized_comments.npy', allow_pickle=True))}")
    print(f"Total number of labels: {len(np.load('data/training_data/test_training_02/pretrained/labels.npy', allow_pickle=True))}")