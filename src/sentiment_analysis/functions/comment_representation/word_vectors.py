


import numpy as np


def create_word_vectors(
        word_list:list,
        word_vector_size:int
    ):
    """
    Create a dictionary of word vectors for the given list of words.
    
    Args:
        1. word_list (list): A list of words to create vectors for.
        2. word_vector_size (int): The size of each word vector.
    
    The Function:
        1. Iterates through each word in the word_list.
        2. Generates a random vector of size word_vector_size for each word.
        3. Stores the word and its corresponding vector in a dictionary.
        4. Returns the dictionary of word vectors.

    Returns:
        1. word_vectors: A dictionary mapping each word to its corresponding vector.
    """
    
    import numpy as np
    

    # Can this be done with "list" comprehension?

    word_vectors = {}
    
    for word in word_list:
        # Generate a random vector for each word.
        vector = np.random.rand(word_vector_size)
        word_vectors[word] = vector
    
    return word_vectors


# Example usage:
if __name__ == "__main__":
    words = ["apple", "banana", "cherry"]
    vector_size = 5
    vectors = create_word_vectors(words, vector_size)
    
    for word, vector in vectors.items():
        print(f"Word: {word}, Vector: {vector}")